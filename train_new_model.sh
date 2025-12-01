#!/bin/bash
# Yeni Keypoint'lerle Model Eğitimi
# Normal dataset'ine yeni keypoint'ler eklendikten sonra çalıştırılacak

echo "🚀 YENİ MODEL EĞİTİMİ BAŞLIYOR..."
echo "============================================================"

# Model adı (istediğiniz ismi verebilirsiniz)
MODEL_NAME="new_model"
SEED=1111  # En iyi seed (dataset_improved'da kullanılan)

echo ""
echo "📊 Dataset kontrol ediliyor..."
echo "============================================================"
python3 -c "
import json
import os

metadata_path = 'keypoints/metadata.json'
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    normal_count = sum(1 for item in metadata if 'normal' in item.get('video_path', '').lower())
    scoliosis_count = sum(1 for item in metadata if 'scoliosis' in item.get('video_path', '').lower())
    
    print(f'✅ Toplam keypoint: {len(metadata)}')
    print(f'   Normal: {normal_count}')
    print(f'   Scoliosis: {scoliosis_count}')
else:
    print('❌ metadata.json bulunamadı!')
    exit(1)
"

echo ""
echo "🎯 Model eğitiliyor..."
echo "============================================================"
echo "   Model: advanced_lstm (attention ON)"
echo "   Seed: $SEED"
echo "   Epochs: 100"
echo "   Learning Rate: 0.001"
echo "   Batch Size: 16"
echo "============================================================"

python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --save_dir saved_models \
    --model_name $MODEL_NAME \
    --seed $SEED \
    --device cuda

# Test setinde değerlendir
echo ""
echo "📈 Test setinde değerlendiriliyor..."
echo "============================================================"
python3 src/evaluate.py \
    --model_path saved_models/${MODEL_NAME}.pth \
    --model_type advanced_lstm \
    --keypoints_dir keypoints \
    --device cuda

echo ""
echo "✅ EĞİTİM TAMAMLANDI!"
echo "============================================================"
echo "Model kaydedildi: saved_models/${MODEL_NAME}.pth"
echo ""
echo "📊 Sonuçları görmek için:"
echo "   cat results/evaluation_metrics.json"

