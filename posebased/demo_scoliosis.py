#!/usr/bin/env python3
"""
Skolyoz Analizi - Örnek Kullanım Scripti
Bu script, scoliosis_analysis.py dosyasının nasıl kullanılacağını gösterir.
"""

import os
import sys
from pathlib import Path

# Ana script'i import et
from scoliosis_analysis import ScoliosisAnalyzer


def demo_training():
    """Demo eğitim örneği - Helin'in videoları ile"""
    print("=== HELİN'İN VİDEOLARI İLE EĞİTİM ===")
    
    # Helin'in gerçek video dosyaları
    video_paths = [
        # Normal videolar
        "../my_videos/normal/kamera2_20251005_124448.avi",
        "../my_videos/normal/kamera2_20251005_125528.avi", 
        "../my_videos/normal/kamera2_20251005_134659.avi",
        # Skolyoz videolar
        "../my_videos/scoliosis/hasta_kamera2_20251005_114655.avi",
        "../my_videos/scoliosis/kamera2_20251005_122212.avi",
        "../my_videos/scoliosis/kamera2_20251005_133745.avi"
    ]
    
    # Etiketler (0: Normal, 1: Skolyoz)
    labels = [0, 0, 0, 1, 1, 1]
    
    # Mevcut olmayan dosyaları kontrol et
    existing_videos = []
    existing_labels = []
    
    for video_path, label in zip(video_paths, labels):
        if os.path.exists(video_path):
            existing_videos.append(video_path)
            existing_labels.append(label)
        else:
            print(f"Uyarı: {video_path} bulunamadı")
    
    if len(existing_videos) < 2:
        print("Hata: En az 2 video dosyası gerekli!")
        print("Lütfen demo_videos klasörüne video dosyalarınızı ekleyin.")
        return
    
    # Analyzer oluştur
    analyzer = ScoliosisAnalyzer(model_type="lstm")
    
    try:
        # Dataset hazırla
        train_loader, test_loader = analyzer.prepare_dataset(existing_videos, existing_labels)
        
        # Model eğit (daha uzun eğitim - Helin'in videoları için)
        train_losses, test_accuracies = analyzer.train_model(
            train_loader, test_loader, epochs=30
        )
        
        # Model kaydet
        analyzer.save_model("scoliosis_model_demo.pth")
        
        print("\n=== EĞİTİM BAŞARILI ===")
        print("Model 'scoliosis_model_demo.pth' olarak kaydedildi")
        
    except Exception as e:
        print(f"Eğitim hatası: {e}")


def demo_prediction():
    """Demo tahmin örneği"""
    print("\n=== DEMO TAHMİN ===")
    
    # Model dosyası var mı kontrol et
    model_path = "scoliosis_model_demo.pth"
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        print("Önce demo_training() çalıştırın.")
        return
    
    # Test video dosyası - Helin'in videolarından birini kullan
    test_video = "../my_videos/normal/kamera2_20251005_124448.avi"
    if not os.path.exists(test_video):
        print(f"Uyarı: Test video bulunamadı: {test_video}")
        print("Lütfen test video dosyasını ekleyin.")
        return
    
    try:
        # Analyzer oluştur ve model yükle
        analyzer = ScoliosisAnalyzer(model_type="lstm")
        analyzer.load_model(model_path)
        
        # Tahmin yap
        result = analyzer.predict_video(test_video)
        
        print("\n=== TAHMİN SONUCU ===")
        print(f"Video: {result['video_path']}")
        print(f"Tahmin: {result['prediction']}")
        print(f"Güven Skorları:")
        print(f"  Normal: {result['confidence']['Normal']:.3f}")
        print(f"  Skolyoz: {result['confidence']['Skolyoz']:.3f}")
        
    except Exception as e:
        print(f"Tahmin hatası: {e}")


def create_demo_structure():
    """Demo için klasör yapısı oluştur"""
    print("=== DEMO KLASÖR YAPISI OLUŞTURULUYOR ===")
    
    # Demo klasörlerini oluştur
    demo_dirs = [
        "datasets/demo_videos",
        "models",
        "results"
    ]
    
    for demo_dir in demo_dirs:
        Path(demo_dir).mkdir(parents=True, exist_ok=True)
        print(f"Klasör oluşturuldu: {demo_dir}")
    
    # README dosyası oluştur
    readme_content = """# Skolyoz Analizi Demo

Bu klasör skolyoz analizi için demo video dosyalarını içerir.

## Klasör Yapısı
- `normal_1.mp4`, `normal_2.mp4`: Normal duruş videoları
- `scoliosis_1.mp4`, `scoliosis_2.mp4`: Skolyoz duruş videoları  
- `test_video.mp4`: Test için video

## Video Formatları
Desteklenen formatlar: MP4, AVI, MOV, MKV

## Video Önerileri
- Yan profil çekim (90 derece açı)
- Kişi ayakta durmalı
- Minimum 5-10 saniye uzunluk
- Net görüntü kalitesi
- Tek kişi görüntüde olmalı
"""
    
    with open("datasets/demo_videos/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("README.md dosyası oluşturuldu: datasets/demo_videos/README.md")


def main():
    """Ana demo fonksiyonu"""
    print("Skolyoz Analizi - Demo Script")
    print("=" * 40)
    
    while True:
        print("\nSeçenekler:")
        print("1. Demo klasör yapısı oluştur")
        print("2. Demo eğitim çalıştır")
        print("3. Demo tahmin çalıştır")
        print("4. Çıkış")
        
        choice = input("\nSeçiminizi yapın (1-4): ").strip()
        
        if choice == "1":
            create_demo_structure()
        elif choice == "2":
            demo_training()
        elif choice == "3":
            demo_prediction()
        elif choice == "4":
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim! Lütfen 1-4 arası bir sayı girin.")


if __name__ == "__main__":
    main()
