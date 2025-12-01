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

# Proje root dizinini bul
def get_project_root():
    """Proje root dizinini bul"""
    current_file = os.path.abspath(__file__)
    # src/visualize_gui.py -> proje root
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

# En iyi modeli otomatik yükle
def auto_load_best_model():
    """En iyi modeli otomatik yükle (dataset_improved.pth)"""
    global model, extractor, device
    
    # Proje root dizinini bul
    project_root = get_project_root()
    model_path = os.path.join(project_root, "saved_models", "dataset_improved.pth")
    
    # Alternatif path'leri dene
    possible_paths = [
        model_path,  # Proje root'tan
        "saved_models/dataset_improved.pth",  # Mevcut dizinden
        os.path.join(os.getcwd(), "saved_models", "dataset_improved.pth"),  # Çalışma dizininden
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        # Tüm modelleri listele
        saved_models_dir = os.path.join(project_root, "saved_models")
        available_models = []
        if os.path.exists(saved_models_dir):
            available_models = [f for f in os.listdir(saved_models_dir) if f.endswith('.pth')]
        
        models_list = "\n".join([f"  - {m}" for m in available_models]) if available_models else "  (none found)"
        return False, f"❌ Best model not found!\n\nSearched paths:\n  - {os.path.join(project_root, 'saved_models', 'dataset_improved.pth')}\n  - saved_models/dataset_improved.pth\n  - {os.path.join(os.getcwd(), 'saved_models', 'dataset_improved.pth')}\n\nAvailable models in saved_models/:\n{models_list}\n\nCurrent directory: {os.getcwd()}\nProject root: {project_root}"
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # AdvancedLSTM with attention
        model = build_model(model_type="advanced_lstm", use_attention=True)
        model = model.to(device)
        load_checkpoint(model_path, model)
        model.eval()
        
        extractor = PoseExtractor()
        
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        return True, f"✅ Best model loaded automatically!\n🧠 Model: dataset_improved (81.82% accuracy)\n📁 Path: {model_path}\n📱 Device: {gpu_info}"
    except Exception as e:
        return False, f"❌ Model loading error: {str(e)}\n\nTried path: {model_path}"


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


def process_video(video_file):
    """Process video, draw keypoints and make prediction"""
    global model
    
    if video_file is None:
        return None, "", "⚠️ Please upload a video file!"
    
    # Model yüklü mü kontrol et
    if model is None:
        success, msg = auto_load_best_model()
        if not success:
            return None, "", f"❌ {msg}\n\nPlease ensure saved_models/dataset_improved.pth exists."
    
    try:
        # Gradio file upload formatını handle et
        if isinstance(video_file, str):
            video_path = video_file
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        elif isinstance(video_file, dict) and 'name' in video_file:
            video_path = video_file['name']
        else:
            video_path = str(video_file)
        
        if not os.path.exists(video_path):
            return None, "", f"❌ Video file not found: {video_path}"
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "", f"❌ Video could not be opened: {video_path}"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:  # NaN kontrolü
            fps = 30.0  # Varsayılan FPS
        fps = int(fps)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width <= 0 or height <= 0:
            return None, "", f"❌ Invalid video dimensions: {width}x{height}"
        
        # Keypoint'leri canlı olarak çıkar
        all_keypoints_list = []
        
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Çıkış video - kalıcı klasöre kaydet
        project_root = get_project_root()
        saved_videos_dir = os.path.join(project_root, "saved_videos")
        os.makedirs(saved_videos_dir, exist_ok=True)
        
        # Benzersiz dosya adı oluştur
        import time
        timestamp = int(time.time())
        video_name = f"analysis_{timestamp}_{os.getpid()}.mp4"
        output_path = os.path.join(saved_videos_dir, video_name)
        
        # Video codec'leri sırayla dene
        codecs_to_try = [
            ('avc1', 'H.264'),
            ('mp4v', 'MPEG-4'),
            ('XVID', 'XVID'),
            ('MJPG', 'Motion JPEG')
        ]
        
        out = None
        used_codec = None
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    used_codec = codec_desc
                    break
                else:
                    out.release()
            except:
                continue
        
        if out is None or not out.isOpened():
            return None, "", f"❌ Video codec could not be opened! Tried: {', '.join([c[1] for c in codecs_to_try])}"
        
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
            
            # Canlı keypoint çıkarımı
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
            
            color = (0, 255, 0) if has_pose else (0, 0, 255)
            text = "POSE DETECTED" if has_pose else "NO POSE"
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated_frame, text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            out.write(annotated_frame)
            frame_count += 1
        
        cap.release()
        if out:
            out.release()
        if pose:
            pose.close()
        
        # Video dosyasının oluşturulduğundan emin ol
        import time
        max_wait = 10  # Maksimum 10 saniye bekle
        wait_time = 0
        while not os.path.exists(output_path) and wait_time < max_wait:
            time.sleep(0.2)
            wait_time += 0.2
        
        if not os.path.exists(output_path):
            return None, "", f"❌ Video file could not be created: {output_path}"
        
        # Dosyanın tamamen yazıldığından emin ol (dosya boyutu sabit kalana kadar bekle)
        prev_size = 0
        stable_count = 0
        for _ in range(20):  # Maksimum 2 saniye bekle
            time.sleep(0.1)
            if os.path.exists(output_path):
                current_size = os.path.getsize(output_path)
                if current_size == prev_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 3:  # 3 kez aynı boyutta ise tamamlanmış demektir
                        break
                else:
                    stable_count = 0
                prev_size = current_size
        
        # Mutlak path'e çevir (Gradio için gerekli)
        output_path = os.path.abspath(output_path)
        
        # Dosyanın okunabilir olduğundan emin ol
        if not os.access(output_path, os.R_OK):
            return None, "", f"❌ Video file is not readable: {output_path}"
        
        detection_rate = (detected_frames / frame_count) * 100 if frame_count > 0 else 0
        
        # Tahmin yap (her zaman yapılır)
        prediction_result = ""
        prediction_label = ""
        normal_prob = 0
        scoliosis_prob = 0
        confidence = 0
        
        if model is not None and len(all_keypoints_list) > 0:
            keypoints_array = np.array(all_keypoints_list)
            prediction_label, confidence, normal_prob, scoliosis_prob, pred_msg = predict_skoliosis(keypoints_array)
            prediction_result = "completed"
        
        info = f"## ✅ Analysis Completed!\n\n"
        info += f"### 📹 Video Information\n"
        info += f"- **Total frames:** {frame_count}\n"
        info += f"- **Pose detected:** {detected_frames} frames\n"
        info += f"- **Detection rate:** {detection_rate:.1f}%\n"
        info += f"- **Video resolution:** {width}x{height}\n"
        info += f"- **FPS:** {fps}\n\n"
        
        if prediction_result == "completed" and prediction_label:
            info += f"### 🎯 AI Analysis Result\n"
            info += f"- **Prediction:** **{prediction_label}**\n"
            info += f"- **Confidence:** {confidence:.1f}%\n"
            info += f"- **Normal probability:** {normal_prob:.1f}%\n"
            info += f"- **Scoliosis probability:** {scoliosis_prob:.1f}%\n"
            info += f"\n### 📊 Model Information\n"
            info += f"- **Model:** dataset_improved (Best Model)\n"
            info += f"- **Accuracy:** 81.82%\n"
            info += f"- **Precision:** 86.36%\n"
            info += f"- **Recall:** 81.82%\n"
        
        # HTML format prediction result
        prediction_html = ""
        if prediction_result == "completed" and prediction_label:
            if "Uncertain" in prediction_label:
                prediction_html = f"""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); color: #856404; padding: 25px; border-radius: 12px; border: 3px solid #ffc107; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 3em; margin-bottom: 15px;">⚠️</div>
                    <h2 style="margin: 0 0 15px 0; font-size: 1.8em; font-weight: bold;">Uncertain Result</h2>
                    <p style="margin: 10px 0; font-size: 1.1em;"><strong>Confidence Level:</strong> {confidence:.1f}%</p>
                    <div style="display: flex; justify-content: space-around; margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.3); border-radius: 8px;">
                        <div>
                            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">Normal</p>
                            <p style="margin: 5px 0; font-size: 1.5em; font-weight: bold;">{normal_prob:.1f}%</p>
                        </div>
                        <div>
                            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">Scoliosis</p>
                            <p style="margin: 5px 0; font-size: 1.5em; font-weight: bold;">{scoliosis_prob:.1f}%</p>
                        </div>
                    </div>
                    <p style="margin: 15px 0 0 0; font-size: 0.95em; font-style: italic;">Model confidence is too low. Please try with a clearer video.</p>
                </div>
                """
            elif prediction_label == "Normal":
                prediction_html = f"""
                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); color: #155724; padding: 25px; border-radius: 12px; border: 3px solid #28a745; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 3em; margin-bottom: 15px;">✅</div>
                    <h2 style="margin: 0 0 15px 0; font-size: 1.8em; font-weight: bold;">Normal Posture</h2>
                    <p style="margin: 10px 0; font-size: 1.1em;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <div style="display: flex; justify-content: space-around; margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.3); border-radius: 8px;">
                        <div>
                            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">Normal</p>
                            <p style="margin: 5px 0; font-size: 1.5em; font-weight: bold;">{normal_prob:.1f}%</p>
                        </div>
                        <div>
                            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">Scoliosis</p>
                            <p style="margin: 5px 0; font-size: 1.5em; font-weight: bold;">{scoliosis_prob:.1f}%</p>
                        </div>
                    </div>
                    <p style="margin: 15px 0 0 0; font-size: 0.95em; color: #155724;">No signs of scoliosis detected in the analysis.</p>
                </div>
                """
            else:
                prediction_html = f"""
                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); color: #721c24; padding: 25px; border-radius: 12px; border: 3px solid #dc3545; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 3em; margin-bottom: 15px;">⚠️</div>
                    <h2 style="margin: 0 0 15px 0; font-size: 1.8em; font-weight: bold;">Scoliosis Detected</h2>
                    <p style="margin: 10px 0; font-size: 1.1em;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <div style="display: flex; justify-content: space-around; margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.3); border-radius: 8px;">
                        <div>
                            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">Normal</p>
                            <p style="margin: 5px 0; font-size: 1.5em; font-weight: bold;">{normal_prob:.1f}%</p>
                        </div>
                        <div>
                            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">Scoliosis</p>
                            <p style="margin: 5px 0; font-size: 1.5em; font-weight: bold;">{scoliosis_prob:.1f}%</p>
                        </div>
                    </div>
                    <p style="margin: 15px 0 0 0; font-size: 0.95em; color: #721c24; font-weight: bold;">⚠️ Please consult with a medical professional for proper diagnosis.</p>
                </div>
                """
        
        # Video path'in mutlak path olduğundan ve dosyanın var olduğundan emin ol
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        
        # Dosya boyutunu kontrol et (boş dosya olmamalı)
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            return None, "", "❌ Video file is empty!"
        
        return output_path, prediction_html, info
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"❌ Error occurred during video processing:\n\n{str(e)}\n\nDetails:\n{error_details}"
        print(f"ERROR in process_video: {error_msg}")  # Console'a da yazdır
        return None, "", error_msg


# Gradio UI - Profesyonel tasarım
with gr.Blocks(title="Scoliosis Analysis System") as demo:
    gr.HTML("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; color: white; font-size: 2.5em; font-weight: bold;">🏥 Scoliosis Analysis System</h1>
        <p style="margin: 15px 0 0 0; font-size: 1.2em; opacity: 0.95;">AI-Powered Posture Analysis & Scoliosis Detection</p>
        <p style="margin: 10px 0 0 0; font-size: 0.9em; opacity: 0.85;">Upload video → Extract keypoints → AI analysis → Results</p>
    </div>
    """)
    
    # Model durumu göster
    model_status_display = gr.HTML()
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📤 Upload Video
            
            Upload a video file to analyze. The system will:
            - Extract pose keypoints automatically
            - Visualize keypoints on video
            - Perform AI-powered scoliosis analysis
            """)
            
            video_input = gr.File(
                label="📹 Select Video File",
                file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv"],
                height=100
            )
            
            process_btn = gr.Button(
                "🚀 Analyze Video", 
                variant="primary",
                size="lg",
                scale=2
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Analysis Results")
            
            video_output = gr.Video(
                label="📹 Video with Keypoints Visualization",
                height=400,
                autoplay=True
            )
            
            prediction_output = gr.HTML(
                label="🎯 Scoliosis Analysis Result",
                elem_classes=["prediction-box"]
            )
            
            with gr.Accordion("📊 Detailed Information", open=False):
                info_output = gr.Markdown()
    
    gr.Markdown("""
    ---
    ### 📝 How to Use
    
    1. **Upload Video**: Click "Select Video File" and choose your video
    2. **Analyze**: Click "Analyze Video" button
    3. **View Results**: 
       - Watch the video with keypoints visualized
       - See the AI analysis result (Normal or Scoliosis)
       - Check detailed statistics
    
    ### 🎯 Analysis Information
    
    - **Model**: Best performing model (81.82% accuracy)
    - **Keypoints**: 33 body keypoints extracted using MediaPipe
    - **Analysis**: Real-time AI prediction with confidence scores
    
    ### ⚠️ Important Notes
    
    - Video should show person from front view
    - Ensure good lighting and clear visibility
    - Results are for reference only - consult a medical professional for diagnosis
    """)
    
    # Başlangıçta modeli yükle
    def initialize_model():
        success, msg = auto_load_best_model()
        if success:
            return f"""
            <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 2px solid #28a745; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0;">✅ {msg}</h4>
            </div>
            """
        else:
            return f"""
            <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 2px solid #dc3545; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0;">❌ {msg}</h4>
            </div>
            """
    
    # Video işleme
    process_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[video_output, prediction_output, info_output]
    )
    
    # Sayfa yüklendiğinde modeli yükle
    demo.load(
        fn=initialize_model,
        outputs=[model_status_display]
    )


if __name__ == "__main__":
    print("🌐 Scoliosis Analysis Web UI starting...")
    print("📱 Will open in browser: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
