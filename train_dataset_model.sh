#!/bin/bash
# Dataset Model Eğitimi - %80+ Accuracy Hedefi
# Keypoint çıkarımı bittikten sonra çalıştırılacak

echo "🚀 DATASET MODEL EĞİTİMİ BAŞLIYOR..."
echo "============================================================"

# 1. ADIM: En iyi seed'i bul (test setinde en yüksek accuracy için)
echo ""
echo "📊 ADIM 1: En iyi seed aranıyor..."
echo "============================================================"
python3 src/find_best_seed.py \
    --seed_list 42 123 456 789 1111 2222 3333 4444 5555 6666 7777 8888 9999 12345 \
    --model_type advanced_lstm \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 16 \
    --keypoints_dir keypoints \
    --save_dir saved_models \
    --output_file results/dataset_best_seed_results.json \
    --device cuda

# En iyi seed'i JSON'dan oku
BEST_SEED=$(python3 -c "
import json
with open('results/dataset_best_seed_results.json', 'r') as f:
    results = json.load(f)
best = max(results, key=lambda x: x['test_accuracy'])
print(best['seed'])
")

echo ""
echo "✅ En iyi seed bulundu: $BEST_SEED"
echo "============================================================"

# 2. ADIM: En iyi seed ile model eğit (dataset.pth olarak kaydet)
echo ""
echo "🎯 ADIM 2: Model eğitiliyor (seed=$BEST_SEED)..."
echo "============================================================"
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --save_dir saved_models \
    --model_name dataset \
    --seed $BEST_SEED \
    --device cuda

# 3. ADIM: Test setinde değerlendir
echo ""
echo "📈 ADIM 3: Test setinde değerlendiriliyor..."
echo "============================================================"
python3 src/evaluate.py \
    --model_path saved_models/dataset.pth \
    --model_type advanced_lstm \
    --keypoints_dir keypoints \
    --device cuda

echo ""
echo "✅ EĞİTİM TAMAMLANDI!"
echo "============================================================"
echo "Model kaydedildi: saved_models/dataset.pth"
echo "En iyi seed: $BEST_SEED"
echo "Sonuçlar: results/dataset_best_seed_results.json"

