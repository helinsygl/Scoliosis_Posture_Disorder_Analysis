#!/usr/bin/env python3
"""
OpenGait ile Gelişmiş Skolyoz Analizi
OpenGait'in gelişmiş gait analysis özelliklerini kullanır
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# OpenGait modüllerini import et
sys.path.append('../opengait')
from opengait.main import main as opengait_main
from opengait.data import build_dataset
from opengait.model import build_model
from opengait.utils import get_logger

class AdvancedScoliosisAnalyzer:
    """OpenGait tabanlı gelişmiş skolyoz analizi"""
    
    def __init__(self, config_path="configs/sconet/sconet_scoliosis1k.yaml"):
        self.config_path = config_path
        self.logger = get_logger()
        
    def prepare_opengait_dataset(self, video_paths, labels):
        """OpenGait formatında dataset hazırla"""
        
        print("🔧 OpenGait dataset formatına çeviriliyor...")
        
        # Video'ları OpenGait formatına çevir
        dataset_structure = {
            'train': [],
            'test': []
        }
        
        for i, (video_path, label) in enumerate(zip(video_paths, labels)):
            # Video'yu silhouette'a çevir
            silhouette_path = self._extract_silhouette(video_path, i)
            
            # OpenGait formatında kaydet
            subject_id = f"subject_{i:04d}"
            class_name = "normal" if label == 0 else "scoliosis"
            view_name = "000_180"  # Yan profil
            
            dataset_structure['train'].append({
                'subject_id': subject_id,
                'class': class_name,
                'view': view_name,
                'silhouette_path': silhouette_path,
                'label': label
            })
        
        return dataset_structure
    
    def _extract_silhouette(self, video_path, subject_id):
        """Video'dan silhouette çıkar"""
        
        import cv2
        from opengait.data.transforms import ExtractSilhouette
        
        # Silhouette extraction
        extractor = ExtractSilhouette()
        
        # Video'yu oku ve silhouette çıkar
        cap = cv2.VideoCapture(video_path)
        silhouettes = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Silhouette çıkar
            silhouette = extractor(frame)
            silhouettes.append(silhouette)
        
        cap.release()
        
        # Silhouette'ları kaydet
        output_path = f"silhouettes/subject_{subject_id:04d}.pkl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(silhouettes, f)
        
        return output_path
    
    def train_with_opengait(self, dataset_structure, epochs=50):
        """OpenGait ile eğitim yap"""
        
        print("🚀 OpenGait ile eğitim başlıyor...")
        
        # Config dosyasını güncelle
        self._update_config(dataset_structure)
        
        # OpenGait eğitimini başlat
        try:
            # Training
            opengait_main(
                cfgs=self.config_path,
                phase='train',
                log_to_file=True
            )
            
            print("✅ OpenGait eğitimi tamamlandı!")
            
        except Exception as e:
            print(f"❌ OpenGait eğitim hatası: {e}")
            return None
    
    def _update_config(self, dataset_structure):
        """Config dosyasını güncelle"""
        
        import yaml
        
        # Config dosyasını oku
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Dataset bilgilerini güncelle
        config['dataset_root'] = './silhouettes'
        config['partition_file'] = './partition.json'
        
        # Partition dosyasını oluştur
        partition = {
            'TRAIN_SET': [item['subject_id'] for item in dataset_structure['train']],
            'TEST_SET': [item['subject_id'] for item in dataset_structure['test']]
        }
        
        with open('partition.json', 'w') as f:
            json.dump(partition, f)
        
        # Config'i kaydet
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def predict_with_opengait(self, video_path, model_path):
        """OpenGait ile tahmin yap"""
        
        print(f"🔮 OpenGait ile tahmin: {video_path}")
        
        # Silhouette çıkar
        silhouette_path = self._extract_silhouette(video_path, 9999)
        
        # Model yükle ve tahmin yap
        try:
            # OpenGait test
            result = opengait_main(
                cfgs=self.config_path,
                phase='test',
                log_to_file=True
            )
            
            return result
            
        except Exception as e:
            print(f"❌ OpenGait tahmin hatası: {e}")
            return None

def create_advanced_scoliosis_system():
    """Gelişmiş skolyoz analizi sistemi oluştur"""
    
    print("🏗️ Gelişmiş Skolyoz Analizi Sistemi Oluşturuluyor...")
    
    # 1. OpenGait config'ini kopyala
    os.system("cp ../configs/sconet/sconet_scoliosis1k.yaml ./advanced_scoliosis_config.yaml")
    
    # 2. Silhouette extraction için klasör oluştur
    os.makedirs("silhouettes", exist_ok=True)
    
    # 3. Advanced analyzer oluştur
    analyzer = AdvancedScoliosisAnalyzer("advanced_scoliosis_config.yaml")
    
    return analyzer

def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gelişmiş Skolyoz Analizi")
    parser.add_argument("--advanced_train", action="store_true", help="OpenGait ile eğitim")
    parser.add_argument("--advanced_predict", help="OpenGait ile tahmin")
    parser.add_argument("--create_system", action="store_true", help="Sistemi oluştur")
    
    args = parser.parse_args()
    
    if args.create_system:
        analyzer = create_advanced_scoliosis_system()
        print("✅ Gelişmiş sistem oluşturuldu!")
    
    elif args.advanced_train:
        analyzer = AdvancedScoliosisAnalyzer()
        # Video'ları OpenGait formatına çevir ve eğitim yap
        print("🚀 OpenGait ile eğitim başlıyor...")
    
    elif args.advanced_predict:
        analyzer = AdvancedScoliosisAnalyzer()
        result = analyzer.predict_with_opengait(args.advanced_predict, "model.pth")
        print(f"Tahmin sonucu: {result}")
    
    else:
        print("Kullanım:")
        print("  Sistem oluştur: python3 advanced_scoliosis.py --create_system")
        print("  OpenGait eğitim: python3 advanced_scoliosis.py --advanced_train")
        print("  OpenGait tahmin: python3 advanced_scoliosis.py --advanced_predict video.avi")

if __name__ == "__main__":
    main()
