# Skolyoz Analizi - Kullanım Kılavuzu

Bu proje RGB video dosyalarından MediaPipe ile pose keypoint çıkarımı yaparak skolyoz analizi gerçekleştirir.

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements_scoliosis.txt
```

2. Demo klasör yapısını oluşturun:
```bash
python demo_scoliosis.py
# Seçenek 1'i seçin
```

## Kullanım

### 1. Demo ile Başlama

```bash
python demo_scoliosis.py
```

### 2. Manuel Eğitim

```bash
# LSTM modeli ile eğitim
python scoliosis_analysis.py \
    --videos video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
    --labels 0 0 1 1 \
    --model_type lstm \
    --epochs 50 \
    --save_model my_model.pth

# MLP modeli ile eğitim  
python scoliosis_analysis.py \
    --videos video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
    --labels 0 0 1 1 \
    --model_type mlp \
    --epochs 50 \
    --save_model my_model.pth
```

### 3. Tahmin Yapma

```bash
# Eğitilmiş model ile tahmin
python scoliosis_analysis.py \
    --predict test_video.mp4 \
    --load_model my_model.pth
```

## Model Tipleri

- **LSTM**: Sequence verilerini işlemek için ideal, temporal bilgiyi korur
- **MLP**: Daha basit, global average pooling kullanır

## Video Gereksinimleri

- **Format**: MP4, AVI, MOV, MKV
- **Çekim açısı**: Yan profil (90 derece) önerilir
- **Süre**: Minimum 5-10 saniye
- **Kalite**: Net görüntü, tek kişi
- **Pozisyon**: Kişi ayakta durmalı

## Etiketleme

- `0`: Normal duruş
- `1`: Skolyoz duruş

## Çıktı Formatı

```json
{
  "video_path": "test_video.mp4",
  "prediction": "Normal",
  "confidence": {
    "Normal": 0.85,
    "Skolyoz": 0.15
  },
  "raw_probabilities": [0.85, 0.15]
}
```

## Önemli Notlar

- İlk çalıştırmada MediaPipe modeli indirilecek
- GPU varsa otomatik kullanılır
- En az 2 video ile eğitim yapılabilir
- Model ağırlıkları placeholder olarak başlar
- Eğitim sırasında ilerleme gösterilir

## Sorun Giderme

1. **Video açılamıyor**: Format uyumluluğunu kontrol edin
2. **Pose tespit edilemiyor**: Video kalitesini artırın
3. **CUDA hatası**: CPU moduna geçin
4. **Bellek hatası**: Batch size'ı küçültün

## Gelişmiş Kullanım

### Özel Model Parametreleri

```python
from scoliosis_analysis import ScoliosisAnalyzer

# Özel parametrelerle analyzer oluştur
analyzer = ScoliosisAnalyzer(model_type="lstm")

# Özel model parametreleri
model = analyzer.create_model(input_dim=99)
```

### Batch İşleme

```python
# Birden fazla video için tahmin
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = []

for video_path in video_paths:
    result = analyzer.predict_video(video_path)
    results.append(result)
```
