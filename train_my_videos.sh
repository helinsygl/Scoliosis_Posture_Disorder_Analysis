#!/bin/bash
# Skolyoz Analizi - Eğitim Scripti
# Bu script videolarınızı otomatik olarak bulur ve eğitim yapar

echo "=== SKOLYOZ ANALİZİ EĞİTİMİ ==="
echo "Video dosyaları aranıyor..."

# Video dosyalarını bul
NORMAL_VIDEOS=$(find my_videos/normal -name "*.avi" -o -name "*.mp4" -o -name "*.mov" | tr '\n' ' ')
SCOLIOSIS_VIDEOS=$(find my_videos/scoliosis -name "*.avi" -o -name "*.mp4" -o -name "*.mov" | tr '\n' ' ')

# Video sayılarını kontrol et
NORMAL_COUNT=$(echo $NORMAL_VIDEOS | wc -w)
SCOLIOSIS_COUNT=$(echo $SCOLIOSIS_VIDEOS | wc -w)

echo "Bulunan normal videolar: $NORMAL_COUNT"
echo "Bulunan skolyoz videolar: $SCOLIOSIS_COUNT"

if [ $NORMAL_COUNT -eq 0 ] || [ $SCOLIOSIS_COUNT -eq 0 ]; then
    echo "HATA: En az bir normal ve bir skolyoz video gerekli!"
    echo "Lütfen videolarınızı my_videos/normal/ ve my_videos/scoliosis/ klasörlerine koyun."
    exit 1
fi

# Etiketleri oluştur
NORMAL_LABELS=$(printf "0%.0s " $(seq 1 $NORMAL_COUNT))
SCOLIOSIS_LABELS=$(printf "1%.0s " $(seq 1 $SCOLIOSIS_COUNT))

echo ""
echo "=== EĞİTİM BAŞLIYOR ==="
echo "Normal videolar: $NORMAL_VIDEOS"
echo "Skolyoz videolar: $SCOLIOSIS_VIDEOS"
echo ""

# Eğitim komutunu çalıştır
cd posebased
python scoliosis_analysis.py \
    --videos $NORMAL_VIDEOS $SCOLIOSIS_VIDEOS \
    --labels $NORMAL_LABELS $SCOLIOSIS_LABELS \
    --model_type lstm \
    --epochs 50 \
    --save_model ../my_scoliosis_model.pth

echo ""
echo "=== EĞİTİM TAMAMLANDI ==="
echo "Model kaydedildi: my_scoliosis_model.pth"
