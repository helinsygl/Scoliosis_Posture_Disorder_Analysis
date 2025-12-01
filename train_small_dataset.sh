#!/bin/bash
# Küçük Dataset için Optimize Edilmiş Model Eğitimi
# Dataset: 57 video (45 train, 12 validation)
# Overfitting önleme stratejileri uygulanıyor

echo "🚀 KÜÇÜK DATASET İÇİN OPTİMİZE MODEL EĞİTİMİ"
echo "============================================================"
echo "📊 Dataset: 57 video (45 train, 12 validation)"
echo "🎯 Strateji: Overfitting önleme"
echo "============================================================"

# 1. Validation split'i artır (daha fazla validation verisi için)
# test_size=0.3 -> 40 train, 17 validation (daha dengeli)
echo ""
echo "📝 ADIM 1: Dataset split'i optimize ediliyor..."
echo "============================================================"

# Dataset.py'de test_size parametresini geçici olarak değiştirmek için
# Python script ile train edelim

python3 << 'PYTHON_SCRIPT'
import sys
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.append('.')
from src.model import build_model
from src.dataset import KeypointDataset, extract_person_id
from src.utils import set_seed, EarlyStopping, load_checkpoint
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import defaultdict
import json

# Seed sabitle
set_seed(1111)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")

# Dataset yükle - test_size=0.3 ile (daha fazla validation)
keypoints_dir = "keypoints"
metadata_path = os.path.join(keypoints_dir, "metadata.json")

with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

successful = [m for m in metadata if m['status'] == 'success']

keypoint_paths = []
labels = []
person_ids = []

for item in successful:
    video_path = item['video_path']
    keypoint_path = item['keypoint_path']
    
    # Front videoları kullan
    has_side_structure = '/side/' in video_path.lower() or '\\side\\' in video_path.lower()
    has_front_structure = '/front/' in video_path.lower() or '\\front\\' in video_path.lower()
    
    if has_side_structure and not has_front_structure:
        continue
    
    if 'normal' in video_path.lower():
        label = 0
    elif 'scoliosis' in video_path.lower():
        label = 1
    else:
        continue
    
    person_id = extract_person_id(video_path)
    keypoint_paths.append(keypoint_path)
    labels.append(label)
    person_ids.append(person_id)

# Person-based split - test_size=0.3
person_to_indices = defaultdict(list)
for idx, person_id in enumerate(person_ids):
    person_to_indices[person_id].append(idx)

unique_persons = list(person_to_indices.keys())
person_labels = [labels[person_to_indices[pid][0]] for pid in unique_persons]

train_persons, test_persons = train_test_split(
    unique_persons, test_size=0.3,  # 0.2'den 0.3'e çıkarıldı
    random_state=1111, stratify=person_labels
)

train_indices = []
test_indices = []
for pid in train_persons:
    train_indices.extend(person_to_indices[pid])
for pid in test_persons:
    test_indices.extend(person_to_indices[pid])

print(f"👥 Person-based split (test_size=0.3):")
print(f"  Train: {len(train_persons)} kişi, {len(train_indices)} video")
print(f"  Test: {len(test_persons)} kişi, {len(test_indices)} video")

# Dataset oluştur
train_dataset = KeypointDataset(
    [keypoint_paths[i] for i in train_indices],
    [labels[i] for i in train_indices],
    augment=True  # Augmentation açık
)

val_dataset = KeypointDataset(
    [keypoint_paths[i] for i in test_indices],
    [labels[i] for i in test_indices],
    augment=False  # Validation'da augmentation kapalı
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"\n📊 Dataset yüklendi:")
print(f"  Train: {len(train_dataset)} örnek")
print(f"  Validation: {len(val_dataset)} örnek")

# Class weights
train_labels = [labels[i] for i in train_indices]
normal_count = train_labels.count(0)
scoliosis_count = train_labels.count(1)
total = normal_count + scoliosis_count
weight_normal = total / (1.5 * normal_count) if normal_count > 0 else 1.0
weight_scoliosis = total / (2 * scoliosis_count) if scoliosis_count > 0 else 1.0
class_weights = [weight_normal, weight_scoliosis]

print(f"\n📊 Class weights: Normal={weight_normal:.2f}, Scoliosis={weight_scoliosis:.2f}")

# Model - SimpleLSTM kullan (daha basit, overfitting'e daha dayanıklı)
print(f"\n🧠 Model: SimpleLSTM (küçük dataset için optimize)")
model = build_model(model_type="simple_lstm")
model = model.to(device)

# Optimizer - daha fazla weight decay (regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)  # 1e-4'ten 2e-4'e
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, verbose=True
)

weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Early stopping - daha fazla patience
early_stopping = EarlyStopping(patience=30, verbose=True)

# Training
print(f"\n🚀 Eğitim başlıyor...")
print("="*70)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

for epoch in range(100):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()
    
    train_acc = train_correct / train_total
    train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()
    
    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    scheduler.step(val_acc)
    early_stopping(val_acc, model, f"saved_models/new_model_optimized.pth")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, f"saved_models/new_model_optimized.pth")
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/100 - Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}% | Val: Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
    
    if early_stopping.early_stop:
        print("⏹️ Early stopping triggered!")
        break

print(f"\n✅ Eğitim tamamlandı!")
print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
print(f"  Model kaydedildi: saved_models/new_model_optimized.pth")

PYTHON_SCRIPT

echo ""
echo "📈 Test setinde değerlendiriliyor..."
echo "============================================================"
python3 src/evaluate.py \
    --model_path saved_models/new_model_optimized.pth \
    --model_type simple_lstm \
    --keypoints_dir keypoints \
    --device cuda

echo ""
echo "✅ EĞİTİM TAMAMLANDI!"
echo "============================================================"

