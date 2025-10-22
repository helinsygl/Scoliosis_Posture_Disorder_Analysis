#!/usr/bin/env python3
"""
MediaPipe Pose Visualization Scripti
Video üzerinde pose keypoint'lerini çizer ve kaydeder
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visualize_pose_on_video(input_video_path, output_video_path=None):
    """
    Video üzerinde pose keypoint'lerini çizer
    
    Args:
        input_video_path: Giriş video dosyası
        output_video_path: Çıkış video dosyası (opsiyonel)
    """
    
    # Video dosyasını aç
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı: {input_video_path}")
        return
    
    # Video özelliklerini al
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video bilgileri:")
    print(f"  Boyut: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Toplam frame: {total_frames}")
    print(f"  Süre: {total_frames/fps:.1f} saniye")
    
    # Çıkış video writer (eğer belirtilmişse)
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Çıkış video: {output_video_path}")
    
    # MediaPipe pose detection
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_count = 0
        detected_frames = 0
        
        print(f"\nPose detection başlıyor...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # BGR'den RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            results = pose.process(rgb_frame)
            
            # RGB'den BGR'ye geri çevir
            annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Pose landmarks'leri çiz
            if results.pose_landmarks:
                detected_frames += 1
                
                # Pose connections ile çiz
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Frame bilgisi ekle
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "POSE DETECTED", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Pose tespit edilmedi
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, "NO POSE DETECTED", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Çıkış video'ya yaz (eğer belirtilmişse)
            if out:
                out.write(annotated_frame)
            
            # İlerleme göster
            if frame_count % 30 == 0:
                detection_rate = (detected_frames / frame_count) * 100
                print(f"İşlenen frame: {frame_count}/{total_frames} ({detection_rate:.1f}% pose detected)")
    
    # Temizlik
    cap.release()
    if out:
        out.release()
    
    # Sonuçları göster
    detection_rate = (detected_frames / frame_count) * 100
    print(f"\n=== SONUÇLAR ===")
    print(f"Toplam frame: {frame_count}")
    print(f"Pose tespit edilen frame: {detected_frames}")
    print(f"Tespit oranı: {detection_rate:.1f}%")
    
    if output_video_path:
        print(f"Çıkış video kaydedildi: {output_video_path}")

def show_keypoint_info():
    """MediaPipe pose keypoint bilgilerini göster"""
    
    print("=== MEDIAPIPE POSE KEYPOINTS ===")
    print("Toplam 33 keypoint:")
    
    # Keypoint isimleri ve açıklamaları
    keypoints = [
        ("0", "Nose", "Burun"),
        ("1", "Left eye inner", "Sol göz iç"),
        ("2", "Left eye", "Sol göz"),
        ("3", "Left eye outer", "Sol göz dış"),
        ("4", "Right eye inner", "Sağ göz iç"),
        ("5", "Right eye", "Sağ göz"),
        ("6", "Right eye outer", "Sağ göz dış"),
        ("7", "Left ear", "Sol kulak"),
        ("8", "Right ear", "Sağ kulak"),
        ("9", "Mouth left", "Sol ağız"),
        ("10", "Mouth right", "Sağ ağız"),
        ("11", "Left shoulder", "Sol omuz"),
        ("12", "Right shoulder", "Sağ omuz"),
        ("13", "Left elbow", "Sol dirsek"),
        ("14", "Right elbow", "Sağ dirsek"),
        ("15", "Left wrist", "Sol bilek"),
        ("16", "Right wrist", "Sağ bilek"),
        ("17", "Left pinky", "Sol serçe parmak"),
        ("18", "Right pinky", "Sağ serçe parmak"),
        ("19", "Left index", "Sol işaret parmak"),
        ("20", "Right index", "Sağ işaret parmak"),
        ("21", "Left thumb", "Sol başparmak"),
        ("22", "Right thumb", "Sağ başparmak"),
        ("23", "Left hip", "Sol kalça"),
        ("24", "Right hip", "Sağ kalça"),
        ("25", "Left knee", "Sol diz"),
        ("26", "Right knee", "Sağ diz"),
        ("27", "Left ankle", "Sol ayak bileği"),
        ("28", "Right ankle", "Sağ ayak bileği"),
        ("29", "Left heel", "Sol topuk"),
        ("30", "Right heel", "Sağ topuk"),
        ("31", "Left foot index", "Sol ayak parmak"),
        ("32", "Right foot index", "Sağ ayak parmak")
    ]
    
    for i, (id, name, desc) in enumerate(keypoints):
        print(f"{id:2s}. {name:20s} - {desc}")
    
    print(f"\nHer keypoint için 3 değer:")
    print(f"  - x: Yatay pozisyon (0-1)")
    print(f"  - y: Dikey pozisyon (0-1)")
    print(f"  - visibility: Görünürlük (0-1)")

def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MediaPipe Pose Visualization")
    parser.add_argument("--input", required=True, help="Giriş video dosyası")
    parser.add_argument("--output", help="Çıkış video dosyası (opsiyonel)")
    parser.add_argument("--info", action="store_true", help="Keypoint bilgilerini göster")
    
    args = parser.parse_args()
    
    if args.info:
        show_keypoint_info()
        return
    
    # Video dosyası var mı kontrol et
    if not os.path.exists(args.input):
        print(f"Hata: Video dosyası bulunamadı: {args.input}")
        return
    
    # Visualization çalıştır
    visualize_pose_on_video(args.input, args.output)

if __name__ == "__main__":
    main()
