#!/usr/bin/env python3
"""
Otomatik Skolyoz Analizi Sistemi
Büyük veri setleri için otomatik video algılama ve eğitim
"""

import os
import glob
import json
import argparse
from pathlib import Path
from scoliosis_analysis import ScoliosisAnalyzer

class AutoScoliosisTrainer:
    """Otomatik skolyoz eğitim sistemi"""
    
    def __init__(self, data_root="my_videos"):
        self.data_root = data_root
        self.supported_formats = ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']
        
    def scan_dataset(self):
        """Veri setini tarar ve videoları kategorilere ayırır"""
        
        print("🔍 Veri seti taranıyor...")
        
        # Klasör yapısını kontrol et
        if not os.path.exists(self.data_root):
            print(f"❌ Veri klasörü bulunamadı: {self.data_root}")
            return None
        
        dataset_info = {
            'normal': [],
            'scoliosis': [],
            'unlabeled': [],
            'total_videos': 0,
            'total_size_gb': 0
        }
        
        # Normal klasörünü tara
        normal_path = os.path.join(self.data_root, 'normal')
        if os.path.exists(normal_path):
            normal_videos = self._find_videos(normal_path)
            dataset_info['normal'] = normal_videos
            print(f"✅ Normal videolar: {len(normal_videos)}")
        
        # Skolyoz klasörünü tara
        scoliosis_path = os.path.join(self.data_root, 'scoliosis')
        if os.path.exists(scoliosis_path):
            scoliosis_videos = self._find_videos(scoliosis_path)
            dataset_info['scoliosis'] = scoliosis_videos
            print(f"✅ Skolyoz videolar: {len(scoliosis_videos)}")
        
        # Etiketlenmemiş videoları tara
        unlabeled_path = os.path.join(self.data_root, 'unlabeled')
        if os.path.exists(unlabeled_path):
            unlabeled_videos = self._find_videos(unlabeled_path)
            dataset_info['unlabeled'] = unlabeled_videos
            print(f"⚠️ Etiketlenmemiş videolar: {len(unlabeled_videos)}")
        
        # Toplam istatistikler
        dataset_info['total_videos'] = len(dataset_info['normal']) + len(dataset_info['scoliosis'])
        
        # Dosya boyutlarını hesapla
        total_size = 0
        for video_list in [dataset_info['normal'], dataset_info['scoliosis'], dataset_info['unlabeled']]:
            for video in video_list:
                if os.path.exists(video):
                    total_size += os.path.getsize(video)
        
        dataset_info['total_size_gb'] = total_size / (1024**3)
        
        return dataset_info
    
    def _find_videos(self, directory):
        """Belirtilen klasördeki tüm video dosyalarını bulur"""
        videos = []
        
        for ext in self.supported_formats:
            pattern = os.path.join(directory, f"**/*{ext}")
            found_videos = glob.glob(pattern, recursive=True)
            videos.extend(found_videos)
        
        return sorted(videos)
    
    def auto_train(self, model_name="auto_scoliosis_model", epochs=50):
        """Otomatik eğitim yapar"""
        
        print("🚀 Otomatik Eğitim Başlıyor...")
        
        # Veri setini tara
        dataset_info = self.scan_dataset()
        if not dataset_info:
            return None
        
        # Eğitim için yeterli veri var mı kontrol et
        if dataset_info['total_videos'] < 2:
            print("❌ Eğitim için en az 2 video gerekli!")
            return None
        
        if len(dataset_info['normal']) == 0 or len(dataset_info['scoliosis']) == 0:
            print("❌ Hem normal hem skolyoz videolar gerekli!")
            return None
        
        print(f"\n📊 Veri Seti İstatistikleri:")
        print(f"  Normal videolar: {len(dataset_info['normal'])}")
        print(f"  Skolyoz videolar: {len(dataset_info['scoliosis'])}")
        print(f"  Etiketlenmemiş: {len(dataset_info['unlabeled'])}")
        print(f"  Toplam boyut: {dataset_info['total_size_gb']:.2f} GB")
        
        # Video yollarını ve etiketleri hazırla
        video_paths = dataset_info['normal'] + dataset_info['scoliosis']
        labels = [0] * len(dataset_info['normal']) + [1] * len(dataset_info['scoliosis'])
        
        print(f"\n🎯 Eğitim Parametreleri:")
        print(f"  Toplam video: {len(video_paths)}")
        print(f"  Epoch sayısı: {epochs}")
        print(f"  Model adı: {model_name}.pth")
        
        # Analyzer oluştur ve eğitim yap
        analyzer = ScoliosisAnalyzer(model_type="lstm")
        
        try:
            # Dataset hazırla
            train_loader, test_loader = analyzer.prepare_dataset(video_paths, labels)
            
            # Model eğit
            train_losses, test_accuracies = analyzer.train_model(
                train_loader, test_loader, epochs=epochs
            )
            
            # Model kaydet
            model_path = f"{model_name}.pth"
            analyzer.save_model(model_path)
            
            print(f"\n✅ Eğitim Tamamlandı!")
            print(f"  Model kaydedildi: {model_path}")
            print(f"  Son test accuracy: {test_accuracies[-1]:.4f}")
            
            return analyzer, model_path
            
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")
            return None
    
    def auto_predict_batch(self, model_path, test_directory="test_videos"):
        """Toplu tahmin yapar"""
        
        print(f"🔮 Toplu Tahmin Başlıyor...")
        
        if not os.path.exists(test_directory):
            print(f"❌ Test klasörü bulunamadı: {test_directory}")
            return None
        
        # Test videolarını bul
        test_videos = self._find_videos(test_directory)
        
        if not test_videos:
            print(f"❌ Test klasöründe video bulunamadı: {test_directory}")
            return None
        
        print(f"📁 Test videoları: {len(test_videos)}")
        
        # Model yükle
        analyzer = ScoliosisAnalyzer(model_type="lstm")
        analyzer.load_model(model_path)
        
        # Tahmin sonuçları
        results = []
        
        for i, video_path in enumerate(test_videos):
            print(f"\n🎬 Tahmin {i+1}/{len(test_videos)}: {os.path.basename(video_path)}")
            
            try:
                result = analyzer.predict_video(video_path)
                results.append(result)
                
                print(f"  Tahmin: {result['prediction']}")
                print(f"  Güven: Normal={result['confidence']['Normal']:.3f}, Skolyoz={result['confidence']['Skolyoz']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Hata: {e}")
                results.append({"video_path": video_path, "error": str(e)})
        
        # Sonuçları kaydet
        results_file = "batch_predictions.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Toplu tahmin tamamlandı!")
        print(f"  Sonuçlar kaydedildi: {results_file}")
        
        return results
    
    def create_dataset_structure(self):
        """Otomatik veri seti yapısı oluşturur"""
        
        print("📁 Veri seti yapısı oluşturuluyor...")
        
        directories = [
            "my_videos/normal",
            "my_videos/scoliosis", 
            "my_videos/unlabeled",
            "test_videos",
            "models",
            "results"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
        
        # README oluştur
        readme_content = """# Otomatik Skolyoz Analizi Veri Seti

## Klasör Yapısı

- `my_videos/normal/` - Normal duruş videoları
- `my_videos/scoliosis/` - Skolyoz duruş videoları  
- `my_videos/unlabeled/` - Etiketlenmemiş videolar
- `test_videos/` - Test videoları
- `models/` - Eğitilmiş modeller
- `results/` - Sonuçlar

## Desteklenen Formatlar

- AVI, MP4, MOV, MKV, WMV, FLV

## Kullanım

```bash
# Otomatik eğitim
python3 auto_scoliosis.py --auto_train

# Toplu tahmin
python3 auto_scoliosis.py --auto_predict --model models/my_model.pth
```
"""
        
        with open("my_videos/README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print("  ✅ README.md oluşturuldu")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Otomatik Skolyoz Analizi")
    parser.add_argument("--auto_train", action="store_true", help="Otomatik eğitim yap")
    parser.add_argument("--auto_predict", action="store_true", help="Toplu tahmin yap")
    parser.add_argument("--model", help="Model dosyası yolu")
    parser.add_argument("--epochs", type=int, default=50, help="Eğitim epoch sayısı")
    parser.add_argument("--data_root", default="my_videos", help="Veri klasörü")
    parser.add_argument("--test_dir", default="test_videos", help="Test klasörü")
    parser.add_argument("--create_structure", action="store_true", help="Veri seti yapısı oluştur")
    
    args = parser.parse_args()
    
    trainer = AutoScoliosisTrainer(data_root=args.data_root)
    
    if args.create_structure:
        trainer.create_dataset_structure()
    
    elif args.auto_train:
        trainer.auto_train(epochs=args.epochs)
    
    elif args.auto_predict:
        if not args.model:
            print("❌ Model dosyası belirtilmeli: --model model.pth")
            return
        trainer.auto_predict_batch(args.model, args.test_dir)
    
    else:
        print("Kullanım:")
        print("  Otomatik eğitim: python3 auto_scoliosis.py --auto_train")
        print("  Toplu tahmin: python3 auto_scoliosis.py --auto_predict --model model.pth")
        print("  Yapı oluştur: python3 auto_scoliosis.py --create_structure")

if __name__ == "__main__":
    main()
