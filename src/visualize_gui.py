#!/usr/bin/env python3
"""
Skolyoz Analizi Web UI
Video yükleyip keypoint'leri görüntüleyin ve model ile skolyoz analizi yapın
"""

import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
import torch
import sys
from pathlib import Path

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.extract_keypoints import PoseExtractor
from src.utils import load_checkpoint

# MediaPipe setup
mp_pose = mp.solutions.pose

# Global değişkenler
model = None
extractor = None
device = None


def draw_keypoints_on_frame(frame, keypoint_coords):
    """Keypoint'leri frame üzerine çizer"""
    annotated_frame = frame.copy()
    
    # MediaPipe POSE_CONNECTIONS
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Yüz
        (0, 4), (4, 5), (5, 6), (6, 8),  # Yüz
        (9, 10),  # Ağız
        (11, 12),  # Omuzlar
        (11, 13), (13, 15),  # Sol kol
        (12, 14), (14, 16),  # Sağ kol
        (15, 17), (15, 19), (15, 21),  # Sol el
        (16, 18), (16, 20), (16, 22),  # Sağ el
        (11, 23), (12, 24),  # Omuz-kalça
        (23, 24),  # Kalçalar
        (23, 25), (25, 27),  # Sol bacak
        (24, 26), (26, 28),  # Sağ bacak
        (27, 29), (27, 31),  # Sol ayak
        (28, 30), (28, 32),  # Sağ ayak
    ]
    
    # Bağlantıları çiz
    for start_idx, end_idx in connections:
        if start_idx < len(keypoint_coords) and end_idx < len(keypoint_coords):
            start_vis = keypoint_coords[start_idx][2] if len(keypoint_coords[start_idx]) > 2 else 1.0
            end_vis = keypoint_coords[end_idx][2] if len(keypoint_coords[end_idx]) > 2 else 1.0
            
            if start_vis > 0.5 and end_vis > 0.5:
                start_point = (keypoint_coords[start_idx][0], keypoint_coords[start_idx][1])
                end_point = (keypoint_coords[end_idx][0], keypoint_coords[end_idx][1])
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
    
    # Keypoint noktalarını çiz
    for coord in keypoint_coords:
        if len(coord) >= 2:
            x, y = coord[0], coord[1]
            vis = coord[2] if len(coord) > 2 else 1.0
            if vis > 0.5:
                cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
    
    return annotated_frame


def load_model(model_path, model_type="advanced_lstm"):
    """Load model"""
    global model, extractor, device
    
    if not model_path:
        return "⚠️ Please select a model file!"
    
    try:
        model_file = model_path.name if hasattr(model_path, 'name') else model_path
        
        if not os.path.exists(model_file):
            return f"❌ Model file not found: {model_file}"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try loading with selected model type
        try:
            model = build_model(model_type=model_type)
            model = model.to(device)
            load_checkpoint(model_file, model)
        except Exception as e:
            # Try other model types if selected one fails
            model_types = ["simple_lstm", "advanced_lstm", "hybrid", "transformer", "posture"]
            loaded = False
            for mtype in model_types:
                if mtype != model_type:
                    try:
                        model = build_model(model_type=mtype)
                        model = model.to(device)
                        load_checkpoint(model_file, model)
                        model_type = mtype  # Update to the working type
                        loaded = True
                        break
                    except:
                        continue
            
            if not loaded:
                return f"❌ Model loading error: Could not load with any model type. Error: {str(e)}"
        
        model.eval()
        extractor = PoseExtractor()
        
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        return f"✅ Model loaded successfully!\n🧠 Model type: {model_type}\n📱 {gpu_info}"
    except Exception as e:
        return f"❌ Model loading error: {str(e)}"


def predict_skoliosis(keypoints_array, max_sequence_length=100):
    """Predict scoliosis from keypoints"""
    global model, device
    
    if model is None:
        return None, 0, 0, 0, "❌ Model not loaded!"
    
    try:
        # Normalize et (Z-score)
        keypoints = keypoints_array.copy()
        for i in range(0, keypoints.shape[1], 3):
            x_col = keypoints[:, i]
            y_col = keypoints[:, i+1]
            
            if x_col.std() > 1e-8:
                keypoints[:, i] = (x_col - x_col.mean()) / (x_col.std() + 1e-8)
            else:
                keypoints[:, i] = x_col - x_col.mean()
            
            if y_col.std() > 1e-8:
                keypoints[:, i+1] = (y_col - y_col.mean()) / (y_col.std() + 1e-8)
            else:
                keypoints[:, i+1] = y_col - y_col.mean()
        
        # Sequence uzunluğunu sınırla
        if len(keypoints) > max_sequence_length:
            keypoints = keypoints[:max_sequence_length]
        
        # Padding ekle
        if len(keypoints) < max_sequence_length:
            padding_length = max_sequence_length - len(keypoints)
            padding = np.zeros((padding_length, keypoints.shape[1]))
            keypoints = np.vstack([keypoints, padding])
        
        # Tensor'a çevir
        keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
        
        # Tahmin
        with torch.no_grad():
            output = model(keypoints_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
        
        # Sonuçları hazırla
        normal_prob = float(probabilities[0][0].item()) * 100
        scoliosis_prob = float(probabilities[0][1].item()) * 100
        
        # Güven eşiği kontrolü - %65 altındaysa belirsiz
        confidence_threshold = 65.0
        
        if max(normal_prob, scoliosis_prob) < confidence_threshold:
            result = "Uncertain ⚠️"
            confidence = max(normal_prob, scoliosis_prob)
        else:
            result = "Normal" if prediction == 0 else "Scoliosis"
            confidence = normal_prob if prediction == 0 else scoliosis_prob
        
        return result, confidence, normal_prob, scoliosis_prob, "✅ Prediction completed!"
        
    except Exception as e:
        return None, 0, 0, 0, f"❌ Prediction error: {str(e)}"


def process_video(video_file, keypoint_file=None, use_live_detection=True, use_prediction=False):
    """Process video, draw keypoints and make prediction"""
    if video_file is None:
        return None, "", "⚠️ Please select a video!"
    
    try:
        video_path = video_file.name if hasattr(video_file, 'name') else video_file
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "", "❌ Video could not be opened!"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Keypoint'leri yükle veya çıkar
        keypoints = None
        all_keypoints_list = []
        
        pose = None
        if use_live_detection or keypoint_file is None:
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        elif keypoint_file:
            keypoint_path = keypoint_file.name if hasattr(keypoint_file, 'name') else keypoint_file
            if os.path.exists(keypoint_path):
                keypoints = np.load(keypoint_path)
        
        # Çıkış video
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, f"keypoint_video_{os.getpid()}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return None, "", "❌ Video codec could not be opened!"
        
        frame_count = 0
        detected_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = frame.copy()
            has_pose = False
            keypoint_coords = []
            frame_keypoints = []
            
            if use_live_detection and pose:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    has_pose = True
                    detected_frames += 1
                    
                    for landmark in results.pose_landmarks.landmark:
                        x_pixel = int(landmark.x * width)
                        y_pixel = int(landmark.y * height)
                        keypoint_coords.append((x_pixel, y_pixel, landmark.visibility))
                        frame_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                    
                    annotated_frame = draw_keypoints_on_frame(annotated_frame, keypoint_coords)
                else:
                    frame_keypoints = [0.0] * 99
                
                all_keypoints_list.append(frame_keypoints)
            elif keypoints is not None:
                if frame_count < len(keypoints):
                    frame_keypoints_data = keypoints[frame_count]
                    
                    for i in range(0, len(frame_keypoints_data), 3):
                        x_norm = frame_keypoints_data[i]
                        y_norm = frame_keypoints_data[i + 1]
                        visibility = frame_keypoints_data[i + 2]
                        
                        x_pixel = int(x_norm * width)
                        y_pixel = int(y_norm * height)
                        keypoint_coords.append((x_pixel, y_pixel, visibility))
                        
                        if visibility > 0.5:
                            has_pose = True
                    
                    if has_pose:
                        detected_frames += 1
                        annotated_frame = draw_keypoints_on_frame(annotated_frame, keypoint_coords)
            
            color = (0, 255, 0) if has_pose else (0, 0, 255)
            text = "POSE DETECTED" if has_pose else "NO POSE"
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated_frame, text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            out.write(annotated_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        if pose:
            pose.close()
        
        import time
        time.sleep(0.2)
        
        if not os.path.exists(output_path):
            return None, "", "❌ Video file could not be created!"
        
        detection_rate = (detected_frames / frame_count) * 100 if frame_count > 0 else 0
        
        # Tahmin yap
        prediction_result = ""
        prediction_label = ""
        normal_prob = 0
        scoliosis_prob = 0
        confidence = 0
        
        if use_prediction and model is not None:
            if use_live_detection and len(all_keypoints_list) > 0:
                keypoints_array = np.array(all_keypoints_list)
            elif keypoints is not None:
                keypoints_array = keypoints
            else:
                keypoints_array = None
            
            if keypoints_array is not None:
                prediction_label, confidence, normal_prob, scoliosis_prob, pred_msg = predict_skoliosis(keypoints_array)
                prediction_result = "completed"  # Flag to indicate prediction was done
        
        info = f"## ✅ Process Completed!\n\n"
        info += f"### 📹 Video Information\n"
        info += f"- **Total frames:** {frame_count}\n"
        info += f"- **Pose detected:** {detected_frames}\n"
        info += f"- **Detection rate:** {detection_rate:.1f}%\n\n"
        
        if prediction_result == "completed" and prediction_label:
            info += f"### 🎯 Analysis Result\n"
            info += f"- **Prediction:** {prediction_label}\n"
            info += f"- **Confidence:** {confidence:.1f}%\n"
            info += f"- **Normal probability:** {normal_prob:.1f}%\n"
            info += f"- **Scoliosis probability:** {scoliosis_prob:.1f}%\n"
        
        # HTML format prediction result
        prediction_html = ""
        if prediction_result == "completed" and prediction_label:
            if "Uncertain" in prediction_label:
                prediction_html = f"""
                <div style="background: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; border: 2px solid #ffc107; text-align: center;">
                    <h3 style="margin: 0 0 10px 0;">⚠️ Uncertain (Belirsiz)</h3>
                    <p style="margin: 5px 0;"><strong>Confidence too low:</strong> {confidence:.1f}%</p>
                    <p style="margin: 5px 0;"><strong>Normal:</strong> {normal_prob:.1f}% | <strong>Scoliosis:</strong> {scoliosis_prob:.1f}%</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">Model emin değil - daha net bir video deneyin</p>
                </div>
                """
            elif prediction_label == "Normal":
                prediction_html = f"""
                <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 2px solid #28a745; text-align: center;">
                    <h3 style="margin: 0 0 10px 0;">✅ {prediction_label}</h3>
                    <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p style="margin: 5px 0;"><strong>Normal:</strong> {normal_prob:.1f}% | <strong>Scoliosis:</strong> {scoliosis_prob:.1f}%</p>
                </div>
                """
            else:
                prediction_html = f"""
                <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 2px solid #dc3545; text-align: center;">
                    <h3 style="margin: 0 0 10px 0;">⚠️ {prediction_label}</h3>
                    <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p style="margin: 5px 0;"><strong>Normal:</strong> {normal_prob:.1f}% | <strong>Scoliosis:</strong> {scoliosis_prob:.1f}%</p>
                </div>
                """
        
        return output_path, prediction_html, info
        
    except Exception as e:
        return None, "", f"❌ Error: {str(e)}"


# Gradio UI - Güzel tasarım
with gr.Blocks() as demo:
    gr.HTML("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0; color: white;">🏥 Scoliosis Analysis System</h1>
        <p style="margin: 10px 0 0 0;">Upload video, visualize keypoints, and perform AI-powered scoliosis analysis</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            
            model_input = gr.File(
                label="🤖 Model File (.pth)",
                file_types=[".pth"]
            )
            
            model_type_dropdown = gr.Dropdown(
                choices=["advanced_lstm", "simple_lstm", "hybrid", "transformer", "posture"],
                value="advanced_lstm",
                label="🧠 Model Type (advanced_lstm: 80.65%)"
            )
            
            load_model_btn = gr.Button("📥 Load Model", variant="secondary")
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            gr.Markdown("---")
            
            video_input = gr.File(
                label="📹 Video File",
                file_types=[".mp4", ".avi", ".mov", ".mkv"]
            )
            
            keypoint_input = gr.File(
                label="📊 Keypoint File (Optional)",
                file_types=[".npy"]
            )
            
            with gr.Row():
                use_live = gr.Checkbox(
                    label="🔄 Live Keypoint Extraction",
                    value=True
                )
                use_prediction = gr.Checkbox(
                    label="🎯 Perform Scoliosis Analysis",
                    value=True
                )
            
            process_btn = gr.Button("🚀 Process and Analyze", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Results")
            
            video_output = gr.Video(
                label="Video with Keypoints"
            )
            
            with gr.Group():
                prediction_output = gr.HTML(label="🎯 Analysis Result")
            
            with gr.Accordion("📊 Detailed Information"):
                info_output = gr.Markdown()
    
    # Model yükleme
    load_model_btn.click(
        fn=load_model,
        inputs=[model_input, model_type_dropdown],
        outputs=[model_status]
    )
    
    # Video işleme
    process_btn.click(
        fn=process_video,
        inputs=[video_input, keypoint_input, use_live, use_prediction],
        outputs=[video_output, prediction_output, info_output]
    )
    
    gr.Markdown("""
    ---
    ### 📝 User Guide
    
    1. **Load Model**: Select your trained model file (.pth) and click "Load Model" button
    2. **Upload Video**: Select the video file you want to analyze
    3. **Keypoint File** (Optional): Select keypoint file if you have pre-extracted keypoints
    4. **Settings**: 
       - Live keypoint extraction: Use this option if you don't have keypoint file
       - Scoliosis analysis: Makes prediction if model is loaded
    5. **Process**: Click "Process and Analyze" button
    
    ### 🎯 Results
    
    - **Video with Keypoints**: Video displayed with keypoints drawn on it
    - **Analysis Result**: Model prediction (Normal or Scoliosis) and confidence scores
    - **Detailed Information**: Video and analysis statistics
    """)


if __name__ == "__main__":
    print("🌐 Scoliosis Analysis Web UI starting...")
    print("📱 Will open in browser: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
