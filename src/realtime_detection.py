#!/usr/bin/env python3
"""
Real-time Scoliosis Detection
Detects scoliosis/normal from live webcam feed using trained model
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
from collections import deque

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from extract_keypoints import PoseExtractor
from utils import load_checkpoint
import mediapipe as mp

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class RealTimeDetector:
    """Real-time scoliosis detection from webcam"""
    
    def __init__(self, model, extractor, device, window_size=30, update_interval=5):
        """
        Args:
            model: Trained model
            extractor: Pose keypoint extractor
            device: Device (cuda/cpu)
            window_size: Number of frames to accumulate for prediction
            update_interval: Update prediction every N frames
        """
        self.model = model
        self.extractor = extractor
        self.device = device
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Keypoint buffer (sliding window)
        self.keypoint_buffer = deque(maxlen=window_size)
        
        # Current prediction
        self.current_prediction = "Initializing..."
        self.current_confidence = {"Normal": 0.5, "Scoliosis": 0.5}
        self.frame_count = 0
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # MediaPipe pose for visualization
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def normalize_keypoints(self, keypoints):
        """Normalize keypoints using Z-score (same as training)"""
        keypoints = keypoints.copy()
        for i in range(0, keypoints.shape[1], 3):
            # X coordinate
            x_col = keypoints[:, i]
            if x_col.std() > 1e-8:
                keypoints[:, i] = (x_col - x_col.mean()) / (x_col.std() + 1e-8)
            else:
                keypoints[:, i] = x_col - x_col.mean()
            
            # Y coordinate
            y_col = keypoints[:, i+1]
            if y_col.std() > 1e-8:
                keypoints[:, i+1] = (y_col - y_col.mean()) / (y_col.std() + 1e-8)
            else:
                keypoints[:, i+1] = y_col - y_col.mean()
        
        return keypoints
    
    def predict(self, keypoints_array):
        """Make prediction from keypoint sequence"""
        if len(keypoints_array) < 10:  # Need minimum frames
            return None, None
        
        # Normalize
        keypoints_array = self.normalize_keypoints(keypoints_array)
        
        # Convert to tensor
        keypoints_tensor = torch.FloatTensor(keypoints_array).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(keypoints_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
        
        # Get confidence scores
        probs_np = probs.cpu().numpy()[0]
        confidence = {
            "Normal": float(probs_np[0]),
            "Scoliosis": float(probs_np[1])
        }
        
        prediction = "Normal" if pred_class == 0 else "Scoliosis"
        
        return prediction, confidence
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract keypoints
            frame_keypoints = []
            for landmark in results.pose_landmarks.landmark:
                frame_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
            
            self.keypoint_buffer.append(frame_keypoints)
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        else:
            # No pose detected - add zeros
            frame_keypoints = [0.0] * (33 * 3)
            self.keypoint_buffer.append(frame_keypoints)
        
        # Update prediction periodically
        if self.frame_count % self.update_interval == 0 and len(self.keypoint_buffer) >= 10:
            keypoints_array = np.array(list(self.keypoint_buffer))
            prediction, confidence = self.predict(keypoints_array)
            
            if prediction is not None:
                self.current_prediction = prediction
                self.current_confidence = confidence
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-8)
        self.fps_history.append(fps)
        self.last_time = current_time
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Draw information on frame
        self.draw_info(frame, avg_fps)
        
        return frame
    
    def draw_info(self, frame, fps):
        """Draw prediction and info on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Prediction text
        color = (0, 255, 0) if self.current_prediction == "Normal" else (0, 0, 255)
        cv2.putText(frame, f"Prediction: {self.current_prediction}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Confidence scores
        normal_conf = self.current_confidence["Normal"]
        scoliosis_conf = self.current_confidence["Scoliosis"]
        
        cv2.putText(frame, f"Normal: {normal_conf:.1%}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Scoliosis: {scoliosis_conf:.1%}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer status
        buffer_status = f"Buffer: {len(self.keypoint_buffer)}/{self.window_size}"
        cv2.putText(frame, buffer_status, 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", 
                   (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Real-time Scoliosis Detection")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", default="advanced_lstm",
                       choices=["advanced_lstm", "transformer", "hybrid"],
                       help="Model type")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--window_size", type=int, default=30,
                       help="Number of frames to accumulate for prediction (default: 30)")
    parser.add_argument("--update_interval", type=int, default=5,
                       help="Update prediction every N frames (default: 5)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension (auto-detect from checkpoint if not specified)")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of LSTM layers (auto-detect from checkpoint if not specified)")
    parser.add_argument("--dropout", type=float, default=None,
                       help="Dropout rate (auto-detect from checkpoint if not specified)")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load checkpoint and detect model parameters
    print(f"📂 Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Auto-detect model parameters
    if args.hidden_dim is None:
        if 'fc.weight' in state_dict:
            fc_shape = state_dict['fc.weight'].shape
            args.hidden_dim = fc_shape[1] // 2
        elif 'attention_weights.weight' in state_dict:
            attn_shape = state_dict['attention_weights.weight'].shape
            args.hidden_dim = attn_shape[1] // 2
        else:
            args.hidden_dim = 64
    
    if args.num_layers is None:
        lstm_keys = [k for k in state_dict.keys() if 'lstm.weight_ih_l' in k]
        if lstm_keys:
            layer_indices = [int(k.split('_l')[1].split('_')[0]) for k in lstm_keys if '_l' in k]
            args.num_layers = max(layer_indices) + 1 if layer_indices else 1
        else:
            args.num_layers = 1
    
    if args.dropout is None:
        args.dropout = 0.3
    
    print(f"   Hidden Dim: {args.hidden_dim}, Layers: {args.num_layers}, Dropout: {args.dropout}")
    
    # Build model
    if args.model_type == "advanced_lstm":
        model = build_model(
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_attention=True
        )
    else:
        model = build_model(model_type=args.model_type)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.model_path, model)
    model.eval()
    
    # Initialize extractor
    extractor = PoseExtractor()
    
    # Initialize detector
    detector = RealTimeDetector(
        model=model,
        extractor=extractor,
        device=device,
        window_size=args.window_size,
        update_interval=args.update_interval
    )
    
    # Open camera
    print(f"📹 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Real-time detection started!")
    print("   Press 'q' to quit")
    print("   Make sure the person is fully visible in the frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('Real-time Scoliosis Detection', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Real-time detection stopped")


if __name__ == "__main__":
    main()

