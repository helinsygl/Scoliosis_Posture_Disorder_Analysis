#!/usr/bin/env python3
"""
Scoliosis Analysis System
Professional AI-powered posture analysis tool.
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
import shutil
import time

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.extract_keypoints import PoseExtractor
from src.utils import load_checkpoint

# --- Configuration ---
class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
    SAVED_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "saved_videos")
    
    # Visual Styles
    COLOR_PRIMARY = "#4F46E5"  # Indigo-600
    COLOR_SECONDARY = "#EC4899" # Pink-500
    COLOR_SUCCESS = "#10B981"  # Emerald-500
    COLOR_WARNING = "#F59E0B"  # Amber-500
    COLOR_DANGER = "#EF4444"   # Red-500
    
    # Video
    CONFIDENCE_THRESHOLD = 0.5
    PREDICTION_CONFIDENCE_THRESHOLD = 65.0

# --- Core Logic ---
class ScoliosisAnalyzer:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mp_pose = mp.solutions.pose
        self.loaded_model_name = None
        
        # Ensure directories exist
        os.makedirs(Config.SAVED_VIDEOS_DIR, exist_ok=True)
        
    def get_available_models(self):
        """Scans the saved_models directory for .pth files."""
        if not os.path.exists(Config.SAVED_MODELS_DIR):
            return []
        
        models = []
        for f in os.listdir(Config.SAVED_MODELS_DIR):
            if f.endswith(".pth"):
                models.append(f)
        return sorted(models)

    def load_model(self, model_filename):
        """
        Smart Loads a specific model.
        inspects the checkpoint to determine architecture parameters.
        """
        try:
            model_path = os.path.join(Config.SAVED_MODELS_DIR, model_filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # 1. Inspect Checkpoint for Architecture
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Detect hidden_dim from lstm.weight_hh_l0
            # Shape is (4 * hidden_dim, hidden_dim)
            if 'lstm.weight_hh_l0' in state_dict:
                weight_shape = state_dict['lstm.weight_hh_l0'].shape
                hidden_dim = weight_shape[1] 
                
                # Detect num_layers
                num_layers = 0
                while f'lstm.weight_hh_l{num_layers}' in state_dict:
                    num_layers += 1
            else:
                # Fallback default
                hidden_dim = 64
                num_layers = 1

            print(f"🔹 Detected Architecture: Hidden Dim={hidden_dim}, Layers={num_layers}")

            # 2. Re-build Model
            self.model = build_model(
                model_type="advanced_lstm", 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                use_attention=True
            )
            self.model = self.model.to(self.device)
            
            # 3. Load State Dict
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.loaded_model_name = model_filename
            return True, f"Successfully loaded '{model_filename}' (Size: {hidden_dim}x{num_layers})"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to load model: {str(e)}"

    def draw_keypoints(self, frame, keypoint_coords):
        """Draws professional-looking keypoints and skeletons."""
        annotated_frame = frame.copy()
        
        # Connections mapped for visualization
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (27, 31), (28, 30), (28, 32),
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(keypoint_coords) and end_idx < len(keypoint_coords):
                start_pt = keypoint_coords[start_idx]
                end_pt = keypoint_coords[end_idx]
                
                if start_pt[2] > Config.CONFIDENCE_THRESHOLD and end_pt[2] > Config.CONFIDENCE_THRESHOLD:
                    cv2.line(annotated_frame, (start_pt[0], start_pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), 2)

        # Draw points
        for coord in keypoint_coords:
            if coord[2] > Config.CONFIDENCE_THRESHOLD:
                # Outer circle (White border)
                cv2.circle(annotated_frame, (coord[0], coord[1]), 5, (255, 255, 255), -1)
                # Inner circle (Color based on confidence, or fixed)
                cv2.circle(annotated_frame, (coord[0], coord[1]), 3, (0, 255, 0), -1)
                
        return annotated_frame

    def predict(self, keypoints_array, max_sequence_length=100):
        """Runs the prediction on the extracted keypoints sequence."""
        if self.model is None:
            return None
        
        try:
            # Normalization logic
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
            
            # Padding/Truncating
            if len(keypoints) > max_sequence_length:
                keypoints = keypoints[:max_sequence_length]
            elif len(keypoints) < max_sequence_length:
                padding = np.zeros((max_sequence_length - len(keypoints), keypoints.shape[1]))
                keypoints = np.vstack([keypoints, padding])
            
            # Inference
            keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(keypoints_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
            
            normal_prob = float(probabilities[0][0].item()) * 100
            scoliosis_prob = float(probabilities[0][1].item()) * 100
            
            confidence = max(normal_prob, scoliosis_prob)
            is_uncertain = confidence < Config.PREDICTION_CONFIDENCE_THRESHOLD
            
            return {
                "prediction": "Normal" if prediction == 0 else "Scoliosis",
                "normal_prob": normal_prob,
                "scoliosis_prob": scoliosis_prob,
                "confidence": confidence,
                "is_uncertain": is_uncertain
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def process_video(self, video_path, progress_callback=None):
        """Processes the video: extracts keypoints, renders visual, runs prediction."""
        if not video_path:
            raise ValueError("No video provided")
            
        if self.model is None:
            raise ValueError("No model loaded. Please select and load a model first.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        # Video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        
        # Resize for performance (Max height 640p)
        max_height = 640
        if original_height > max_height:
            scale = max_height / original_height
            width = int(original_width * scale)
            height = max_height
            print(f"Resizing video from {original_width}x{original_height} to {width}x{height} for speed.")
        else:
            width = original_width
            height = original_height

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try VP8 (WebM) first for better browser support without ffmpeg, then mp4v
        codecs = ['vp80', 'mp4v', 'avc1'] 
        out = None
        current_codec = ""
        
        for codec in codecs:
            try:
                # VP8 needs .webm extension usually, but .mp4 container might accept it in cv2 depending on backend
                # Safest for browser is .webm for vp80
                ext = '.webm' if codec == 'vp80' else '.mp4'
                
                # We need to recreate temp file with correct extension
                if 'temp_output' in locals() and os.path.exists(temp_output):
                    try: os.unlink(temp_output)
                    except: pass
                temp_output = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Initialized video writer with codec: {codec}")
                    current_codec = codec
                    break
            except Exception as e:
                print(f"Codec {codec} failed: {e}")
                continue
                
        if not out or not out.isOpened():
             raise ValueError("Could not initialize video writer with any supported codec.")

        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        all_keypoints_list = []
        detected_frames = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process Frame
            if width != original_width or height != original_height:
                frame = cv2.resize(frame, (width, height))
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            annotated_frame = frame.copy()
            frame_keypoints = [0.0] * 99
            
            if results.pose_landmarks:
                detected_frames += 1
                keypoint_coords = []
                frame_keypoints = []
                
                for landmark in results.pose_landmarks.landmark:
                    # Visual coords
                    x_px = int(landmark.x * width)
                    y_px = int(landmark.y * height)
                    keypoint_coords.append((x_px, y_px, landmark.visibility))
                    # Data coords
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                
                annotated_frame = self.draw_keypoints(annotated_frame, keypoint_coords)
            
            all_keypoints_list.append(frame_keypoints)
            
            # Overlay Info: Only draw if NOT using webm/vp80 to avoid encoding issues? 
            # Actually cv2 should handle text fine.
            cv2.putText(annotated_frame, f"Analysis Mode | Frame: {frame_idx}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(annotated_frame)
            frame_idx += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        out.release()
        pose.close()

        # Save Final Video
        ext = '.webm' if current_codec == 'vp80' else '.mp4'
        final_video_name = f"analysis_{int(time.time())}{ext}"
        final_path = os.path.join(Config.SAVED_VIDEOS_DIR, final_video_name)
        shutil.copy2(temp_output, final_path)
        os.unlink(temp_output)
        
        # Run Prediction
        keypoints_array = np.array(all_keypoints_list)
        result = self.predict(keypoints_array)
        
        return final_path, result, {
            "total_frames": frame_idx,
            "detected_frames": detected_frames,
            "detection_rate": (detected_frames/frame_idx)*100 if frame_idx > 0 else 0
        }

    def convert_video_for_web(self, video_path):
        """
        Converts AVI/MOV to WebM/MP4 for browser playback if needed.
        Returns path to converted video or original if no conversion needed.
        """
        if not video_path:
            return None
            
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ['.mp4', '.webm']:
            return video_path
            
        # Needs conversion
        print(f"Converting {ext} to web-friendly format...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        
        # Resize if too big (same as process optimization)
        max_height = 640
        if height > max_height:
            scale = max_height / height
            width = int(width * scale)
            height = max_height

        temp_output = tempfile.NamedTemporaryFile(suffix='.webm', delete=False).name
        
        # Try VP8 for best browser support via OpenCV
        try:
            fourcc = cv2.VideoWriter_fourcc(*'vp80')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            if not out.isOpened():
                # Fallback to mp4v
                temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                
            if out.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if frame.shape[0] != height or frame.shape[1] != width:
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                out.release()
                cap.release()
                return temp_output
        except Exception as e:
            print(f"Conversion failed: {e}")
            
        cap.release()
        return video_path

# --- UI Logic & Events ---
analyzer = ScoliosisAnalyzer()



def ui_video_change(video):
    if video is None:
        return "Waiting for video..."
    return "✅ Video Uploaded. Ready to Analyze."

def ui_video_clear():
    return None, "", "Waiting for video...", None

def ui_process_video(video):
    if video is None:
        return None, "", "Please upload a video first."
    
    if analyzer.model is None:
        return None, "", "Error: Model detached. Please go back and reload."
            
    progress = gr.Progress()
    progress(0, desc="Initializing...")
    
    def update_progress(current, total):
        progress(current/total, desc=f"Processing Frame {current}/{total}")
        
    try:
        vid_path = video if isinstance(video, str) else video.name
        output_path, result, stats = analyzer.process_video(vid_path, update_progress)
        
        # Generate HTML Report
        if result is None:
            html = """<div class="result-card error">Analysis Failed. Maybe video has no people?</div>"""
        else:
            result_color = Config.COLOR_SUCCESS if result['prediction'] == 'Normal' else Config.COLOR_DANGER
            if result['is_uncertain']: result_color = Config.COLOR_WARNING
                
            html = f"""
            <div style="background-color: white; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; border: 2px solid {result_color}20;">
                <div style="color: {result_color}; font-size: 3.5rem; margin-bottom: 0.5rem; line-height: 1;">
                    {result['prediction']}
                </div>
                <div style="color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">
                     Confidence: <strong>{result['confidence']:.1f}%</strong>
                </div>
                
                <div style="display: flex; gap: 1rem; justify-content: center;">
                    <div style="background-color: {Config.COLOR_SUCCESS}15; padding: 1rem; border-radius: 0.75rem; min-width: 120px;">
                        <div style="font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">Normal</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {Config.COLOR_SUCCESS};">{result['normal_prob']:.1f}%</div>
                    </div>
                    <div style="background-color: {Config.COLOR_DANGER}15; padding: 1rem; border-radius: 0.75rem; min-width: 120px;">
                        <div style="font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">Scoliosis</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {Config.COLOR_DANGER};">{result['scoliosis_prob']:.1f}%</div>
                    </div>
                </div>
                
                {'<div style="margin-top: 1.5rem; padding: 0.75rem; background-color: #fffbeb; color: #b45309; border-radius: 0.5rem; font-size: 0.85rem; display: inline-block;">⚠️ Low Confidence Result</div>' if result['is_uncertain'] else ''}
            </div>
            """

        stats_md = f"""
        ### 📊 Analysis Statistics
        - **Total Frames**: {stats['total_frames']}
        - **Pose Detected**: {stats['detected_frames']} frames ({stats['detection_rate']:.1f}%)
        """
        
        return output_path, html, stats_md
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"<div style='color:red; padding:1rem;'>Error: {str(e)}</div>", ""

# --- Custom CSS ---
custom_css = """
body { font-family: 'Inter', system-ui, sans-serif; background-color: #f8fafc; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
.header { text-align: center; margin-bottom: 3rem; }
.header h1 { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.header p { color: #64748b; font-size: 1.1rem; }

.wizard-card {
    background: white;
    padding: 3rem;
    border-radius: 1.5rem;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    text-align: center;
    max-width: 600px;
    margin: 0 auto;
}

.primary-btn { 
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important; 
    border: 0 !important; 
    color: white !important; 
    font-weight: 600 !important; 
    font-size: 1.1rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 0.75rem !important;
    transition: transform 0.2s !important;
}
.primary-btn:hover { transform: scale(1.02); }

.secondary-btn {
    background: #f1f5f9 !important;
    color: #475569 !important;
}

/* Hide Tabs for Wizard Flow */
.hidden-tabs > .tab-nav { display: none !important; }
"""


# --- App Layout ---
with gr.Blocks(title="Scoliosis Analysis AI") as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    app_state = gr.State({}) # Store global app state if needed

    with gr.Tabs(selected=0, elem_classes="hidden-tabs") as app_tabs:
        
        # --- Tab 1: Model Selection ---
        with gr.Tab(label="Selection", id=0):
             with gr.Column(elem_classes="container"):
                # Header (Repeated or global? Global is better but let's put it in container)
                gr.HTML("""
                <div class="header">
                    <h1>Scoliosis Analysis AI</h1>
                    <p>Professional Posture Assessment System</p>
                </div>
                """)
                
                gr.HTML("""
                <div class="wizard-card">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
                    <h2 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 2rem; color: #1e293b;">Select AI Model</h2>
                """)
                
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=analyzer.get_available_models(),
                        label="",
                        show_label=False,
                        value=analyzer.get_available_models()[0] if analyzer.get_available_models() else None,
                        interactive=True,
                        container=False
                    )
                    
                    next_btn = gr.Button("Next ➜", variant="primary", elem_classes="primary-btn")
                    
                    error_box = gr.Markdown("", visible=True)
                
                gr.HTML("</div>") # Close card

        # --- Tab 2: Analysis ---
        with gr.Tab(label="Analysis", id=1):
            with gr.Column(elem_classes="container"):
                 # Smaller Header for Analysis Page
                gr.HTML("""
                <div class="header" style="margin-bottom: 2rem;">
                    <h1 style="font-size: 2rem;">Analysis Dashboard</h1>
                </div>
                """)
                
                with gr.Row():
                    back_btn = gr.Button("⬅ Change Model", size="sm", variant="secondary", scale=0)
                
                with gr.Row(equal_height=True):
                    # Left: Input
                    with gr.Column(scale=1):
                        gr.Markdown("### 📹 Upload Video")
                        input_video = gr.Video(label="Input Video")
                        analyze_btn = gr.Button("✨ Run Analysis", variant="primary", elem_classes="primary-btn", size="lg")
                        
                        gr.Markdown("---")
                        stats_output = gr.Markdown("Waiting for video...")

                    # Right: Output
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Results")
                        output_video = gr.Video(label="Processed Output", interactive=False, autoplay=True)
                        prediction_result = gr.HTML(label="Prediction")

    # --- Event Wiring ---
    
    def go_to_analysis_tab(model_name):
        if not model_name:
            return gr.update(), "⚠️ Please select a model first."
        
        success, msg = analyzer.load_model(model_name)
        if not success:
            return gr.update(), f"❌ {msg}"
        
        # Switch to Tab 1 (Analysis)
        return gr.Tabs(selected=1), ""

    def go_back_tab():
        # Switch to Tab 0 (Selection)
        return gr.Tabs(selected=0)

    # Next Button Click
    next_btn.click(
        fn=go_to_analysis_tab,
        inputs=[model_dropdown],
        outputs=[app_tabs, error_box],
        show_progress="hidden"
    )
    
    # Back Button Click
    back_btn.click(
        fn=go_back_tab,
        inputs=[],
        outputs=[app_tabs],
        show_progress="hidden"
    )
    
    # Video Upload Events
    input_video.upload(
        fn=lambda: "⏳ Loading...",
        inputs=None,
        outputs=[stats_output],
        show_progress="hidden"
    ).then(
        fn=ui_video_change,
        inputs=[input_video],
        outputs=[stats_output],
        show_progress="hidden"
    )
    
    # Video Clear Event
    input_video.clear(
        fn=ui_video_clear,
        inputs=[],
        outputs=[output_video, prediction_result, stats_output, input_video],
        show_progress="hidden"
    )

    # Analyze Click
    analyze_btn.click(
        fn=ui_process_video,
        inputs=[input_video],
        outputs=[output_video, prediction_result, stats_output],
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)