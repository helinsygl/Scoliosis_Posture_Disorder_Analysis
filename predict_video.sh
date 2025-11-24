#!/bin/bash
# Skolyoz Analizi - Tahmin Scripti
# Bu script eğitilmiş model ile yeni video tahmini yapar

echo "=== SKOLYOZ ANALİZİ TAHMİNİ ==="

# Model dosyası var mı kontrol et
if [ ! -f "my_scoliosis_model.pth" ]; then
    echo "HATA: Model dosyası bulunamadı: my_scoliosis_model.pth"
    echo "Önce eğitim yapın: ./train_my_videos.sh"
    exit 1
fi

# Tahmin yapılacak video dosyasını al
if [ $# -eq 0 ]; then
    echo "Kullanım: ./predict_video.sh <video_dosyası>"
    echo "Örnek: ./predict_video.sh test_video.avi"
    exit 1
fi

VIDEO_FILE=$1

# Video dosyası var mı kontrol et
if [ ! -f "$VIDEO_FILE" ]; then
    echo "HATA: Video dosyası bulunamadı: $VIDEO_FILE"
    exit 1
fi

echo "Tahmin yapılıyor: $VIDEO_FILE"
echo "Model: my_scoliosis_model.pth"
echo ""

# Tahmin komutunu çalıştır
cd posebased
python scoliosis_analysis.py \
    --predict "../$VIDEO_FILE" \
    --load_model "../my_scoliosis_model.pth"

echo ""
echo "=== TAHMİN TAMAMLANDI ==="
