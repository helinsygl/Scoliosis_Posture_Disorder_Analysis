#!/usr/bin/env python3
"""
MediaPipe ile pose keypoint çıkarımı
Video dosyalarından keypoint'leri çıkarır ve keypoints/ klasörüne kaydeder
"""

import os
import cv2
import numpy as np
import json
import glob
from pathlib import Path
from typing import Optional, Dict
import argparse
from tqdm import tqdm

import mediapipe as mp

# MediaPipe pose detection setup
mp_pose = mp.solutions.pose


class PoseExtractor:
    """MediaPipe kullanarak video karelerinden pose keypoint çıkarımı yapar"""
    
    def __init__(self, model_complexity=1):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Video dosyasından pose keypoint'leri çıkarır
        
        Args:
            video_path: Video dosyasının yolu
            
        Returns:
            keypoints: Shape (num_frames, num_keypoints*3) array
                      Her keypoint için (x, y, visibility) değerleri
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        if not cap.isOpened():
            print(f"Hata: Video dosyası açılamadı: {video_path}")
            return None
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR'den RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 33 keypoint'i çıkar (MediaPipe pose modeli)
                frame_keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                keypoints_list.append(frame_keypoints)
            else:
                # Eğer pose tespit edilemezse sıfırlarla doldur
                frame_keypoints = [0.0] * (33 * 3)  # 33 keypoint * 3 değer
                keypoints_list.append(frame_keypoints)
            
            frame_count += 1
        
        cap.release()
        
        if not keypoints_list:
            print(f"Uyarı: {video_path} dosyasından keypoint çıkarılamadı")
            return None
        
        keypoints_array = np.array(keypoints_list)
        return keypoints_array
    
    def extract_batch(self, video_paths: list, output_dir: str = "keypoints", 
                     incremental: bool = True):
        """
        Toplu video işleme
        
        Args:
            video_paths: Video dosya yolları listesi
            output_dir: Keypoint dosyalarının kaydedileceği klasör
            incremental: Sadece yeni videoları işle (mevcut keypoint'leri atla)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Mevcut metadata'yı yükle (eğer varsa)
        metadata_path = os.path.join(output_dir, "metadata.json")
        existing_results = []
        existing_keypoints = set()
        
        if incremental and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                # Mevcut keypoint dosyalarını set'e ekle
                for item in existing_results:
                    if item.get('status') == 'success' and 'keypoint_path' in item:
                        existing_keypoints.add(item['keypoint_path'])
                        # Video path'i de kontrol et
                        video_name = Path(item['video_path']).stem
                        existing_keypoints.add(os.path.join(output_dir, f"{video_name}.npy"))
                print(f"📂 Mevcut {len(existing_results)} keypoint dosyası bulundu")
            except Exception as e:
                print(f"⚠️ Mevcut metadata yüklenemedi: {e}")
        
        # Yeni videoları filtrele
        new_videos = []
        skipped_count = 0
        
        for video_path in video_paths:
            video_name = Path(video_path).stem
            keypoint_path = os.path.join(output_dir, f"{video_name}.npy")
            
            if incremental and (keypoint_path in existing_keypoints or os.path.exists(keypoint_path)):
                skipped_count += 1
                continue
            
            new_videos.append(video_path)
        
        if incremental and skipped_count > 0:
            print(f"⏭️  {skipped_count} video atlandı (zaten işlenmiş)")
        
        if not new_videos:
            print("✅ Tüm videolar zaten işlenmiş!")
            return existing_results
        
        print(f"🆕 {len(new_videos)} yeni video işlenecek")
        
        # Yeni videoları işle
        new_results = []
        
        for video_path in tqdm(new_videos, desc="Keypoint çıkarımı"):
            keypoints = self.extract_keypoints_from_video(video_path)
            
            if keypoints is not None:
                # Dosya adını oluştur
                video_name = Path(video_path).stem
                output_path = os.path.join(output_dir, f"{video_name}.npy")
                
                # Keypoint'leri kaydet
                np.save(output_path, keypoints)
                
                new_results.append({
                    'video_path': video_path,
                    'keypoint_path': output_path,
                    'num_frames': len(keypoints),
                    'status': 'success'
                })
            else:
                new_results.append({
                    'video_path': video_path,
                    'status': 'failed'
                })
        
        # Mevcut sonuçlarla birleştir
        if incremental:
            # Mevcut sonuçları güncelle (aynı video varsa yeni olanı kullan)
            existing_video_paths = {item['video_path'] for item in existing_results}
            updated_results = [item for item in existing_results 
                             if item['video_path'] not in {r['video_path'] for r in new_results}]
            updated_results.extend(new_results)
            results = updated_results
        else:
            results = new_results
        
        # Metadata kaydet
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        successful_new = len([r for r in new_results if r['status'] == 'success'])
        total_successful = len([r for r in results if r['status'] == 'success'])
        
        print(f"\n✅ {successful_new} yeni video işlendi")
        print(f"📊 Toplam {total_successful} başarılı keypoint dosyası")
        print(f"📁 Keypoint'ler kaydedildi: {output_dir}")
        
        return results


def find_all_videos(dataset_root: str) -> Dict[str, list]:
    """
    Dataset klasöründeki tüm videoları bulur
    Hem eski yapıyı (normal/front/) hem de yeni yapıyı (normal/) destekler
    
    Args:
        dataset_root: Dataset kök klasörü
        
    Returns:
        video_dict: {'normal': {'front': [...], 'side': [...]}, 'scoliosis': {...}}
    """
    video_dict = {
        'normal': {'front': [], 'side': []},
        'scoliosis': {'front': [], 'side': []}
    }
    
    supported_formats = ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']
    
    for class_name in ['normal', 'scoliosis']:
        # Front videoları kontrol et
        front_dir = os.path.join(dataset_root, class_name, 'front')
        if os.path.exists(front_dir):
            for ext in supported_formats:
                pattern = os.path.join(front_dir, f"*{ext}")
                videos = glob.glob(pattern)
                video_dict[class_name]['front'].extend(videos)
        
        # Side videoları kontrol et
        side_dir = os.path.join(dataset_root, class_name, 'side')
        if os.path.exists(side_dir):
            for ext in supported_formats:
                pattern = os.path.join(side_dir, f"*{ext}")
                videos = glob.glob(pattern)
                video_dict[class_name]['side'].extend(videos)
        
        # Eğer front/ klasörü yoksa, direkt normal/ klasörünü kontrol et (yeni yapı)
        if len(video_dict[class_name]['front']) == 0:
            class_dir = os.path.join(dataset_root, class_name)
            if os.path.exists(class_dir):
                for ext in supported_formats:
                    pattern = os.path.join(class_dir, f"*{ext}")
                    videos = glob.glob(pattern)
                    video_dict[class_name]['front'].extend(videos)
    
    return video_dict


def main():
    parser = argparse.ArgumentParser(description="Pose Keypoint Çıkarımı")
    parser.add_argument("--dataset_root", default="dataset", help="Dataset kök klasörü")
    parser.add_argument("--output_dir", default="keypoints", help="Keypoint çıktı klasörü")
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                       help="MediaPipe model complexity (0=light, 1=full, 2=heavy)")
    parser.add_argument("--incremental", action="store_true", default=True,
                       help="Sadece yeni videoları işle (mevcut keypoint'leri atla) - Default: True")
    parser.add_argument("--force", action="store_true",
                       help="Tüm videoları yeniden işle (incremental'ı devre dışı bırak)")
    
    args = parser.parse_args()
    
    # Force mode incremental'ı override eder
    incremental = args.incremental and not args.force
    
    print("🔍 Video dosyaları aranıyor...")
    video_dict = find_all_videos(args.dataset_root)
    
    # Tüm videoları topla - FRONT + SIDE VİDEOLAR
    all_videos = []
    for class_name in ['normal', 'scoliosis']:
        all_videos.extend(video_dict[class_name]['front'])  # Front videolar
        all_videos.extend(video_dict[class_name]['side'])  # Side videolar da dahil
    
    print(f"📹 Toplam {len(all_videos)} video bulundu")
    print(f"  Normal - Front: {len(video_dict['normal']['front'])}")
    print(f"  Normal - Side: {len(video_dict['normal']['side'])}")
    print(f"  Scoliosis - Front: {len(video_dict['scoliosis']['front'])}")
    print(f"  Scoliosis - Side: {len(video_dict['scoliosis']['side'])}")
    
    if not all_videos:
        print("❌ Hiç video bulunamadı!")
        return
    
    if incremental:
        print(f"🔄 Incremental mode: Sadece yeni videolar işlenecek")
    else:
        print(f"🔄 Force mode: Tüm videolar yeniden işlenecek")
    
    # Keypoint çıkarımı
    extractor = PoseExtractor(model_complexity=args.model_complexity)
    extractor.extract_batch(all_videos, args.output_dir, incremental=incremental)


if __name__ == "__main__":
    main()
