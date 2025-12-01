#!/bin/bash
# İyileştirilmiş Model Eğitimi - %80+ Accuracy Hedefi
# Yapılan iyileştirmeler:
# 1. Geliştirilmiş data augmentation (scaling, keypoint dropout)
# 2. Optimize edilmiş class weights (Normal sınıfına 1.5x ağırlık)
# 3. Attention mekanizması aktif

echo "🚀 İYİLEŞTİRİLMİŞ MODEL EĞİTİMİ BAŞLIYOR..."
echo "============================================================"

# En iyi seed'i kullan (1111, 3333, 6666, 8888 hepsi %72.73 gösterdi)
BEST_SEED=1111

echo ""
echo "🎯 Model eğitiliyor (seed=$BEST_SEED, attention=ON)..."
echo "============================================================"

python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --save_dir saved_models \
    --model_name dataset_improved \
    --seed $BEST_SEED \
    --device cuda

# Test setinde değerlendir
echo ""
echo "📈 Test setinde değerlendiriliyor..."
echo "============================================================"
python3 src/evaluate.py \
    --model_path saved_models/dataset_improved.pth \
    --model_type advanced_lstm \
    --keypoints_dir keypoints \
    --device cuda

echo ""
echo "✅ EĞİTİM TAMAMLANDI!"
echo "============================================================"
echo "Model kaydedildi: saved_models/dataset_improved.pth"

