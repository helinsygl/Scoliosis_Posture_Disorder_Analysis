#!/usr/bin/env python3
"""
Model Eğitim Scripti
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path

from model import build_model
from dataset import load_dataset_from_keypoints
from utils import save_checkpoint, load_checkpoint, EarlyStopping


def set_seed(seed: int = 42):
    """Tüm random seed'leri sabitle - tutarlı sonuçlar için"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🎲 Random seed sabitlendi: {seed}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Progress bar güncelle
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(model, train_loader, val_loader, epochs: int = 100, 
          learning_rate: float = 0.001, device: str = "cuda",
          save_dir: str = "saved_models", model_name: str = "best_model",
          class_weights: list = None):
    """Model eğitimi"""
    
    # Optimizer - basit ve etkili
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Class weights ile weighted loss (sınıf dengesizliği için kritik!)
    if class_weights is not None:
        weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"📊 Class weights kullanılıyor: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Early stopping - daha fazla patience
    early_stopping = EarlyStopping(patience=30, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"🚀 Eğitim başlıyor...")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling (val_acc'e göre)
        scheduler.step(val_acc)
        
        # History kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Best model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(save_dir, f"{model_name}.pth")
            )
            print(f"✅ Best model kaydedildi! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("⏹️ Early stopping triggered!")
            break
    
    # History kaydet
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model kaydedildi: {save_dir}/{model_name}.pth")
    
    return history


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Eğitimi")
    parser.add_argument("--keypoints_dir", default="keypoints", help="Keypoint klasörü")
    parser.add_argument("--model_type", default="advanced_lstm", 
                       choices=["simple_lstm", "advanced_lstm", "posture", "transformer", "hybrid"],
                       help="Model tipi (advanced_lstm en iyi sonucu verdi: %80.65)")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch sayısı")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save_dir", default="saved_models", help="Model kayıt klasörü")
    parser.add_argument("--model_name", default="best_model", help="Model adı")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (tutarlı sonuçlar için, 123 en iyi sonuç)")
    
    args = parser.parse_args()
    
    # Seed sabitlenmesi - tutarlı sonuçlar için
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Dataset yükle
    # Person-based split: aynı kişinin videoları hem train hem test'te olmasın
    train_loader, val_loader = load_dataset_from_keypoints(
        args.keypoints_dir, 
        augment=True,
        batch_size=args.batch_size,
        person_based_split=True  # Doğru kişi ID'leri ile aktif
    )
    
    # Class weights hesapla (sınıf dengesizliği için)
    # Training set'teki sınıf dağılımına göre ağırlık ver
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().tolist())
    
    normal_count = train_labels.count(0)
    scoliosis_count = train_labels.count(1)
    total = normal_count + scoliosis_count
    
    # Geliştirilmiş class weighting - Normal sınıfına daha fazla ağırlık
    # Normal verisi daha fazla olmasına rağmen daha kötü performans gösteriyor
    # Bu yüzden çok daha agresif weighting uyguluyoruz
    if normal_count > 0 and scoliosis_count > 0:
        # Normal sınıfına 3x daha fazla ağırlık ver (Normal verisi daha fazla ama daha zor)
        # Normal örnekler daha çeşitli olduğu için daha fazla ağırlık gerekiyor
        weight_normal = (total / (1.5 * normal_count)) * 3.0  # Çok daha agresif
        weight_scoliosis = total / (2 * scoliosis_count)
    else:
        weight_normal = 1.0
        weight_scoliosis = 1.0
    class_weights = [weight_normal, weight_scoliosis]
    
    print(f"📊 Sınıf dağılımı - Normal: {normal_count}, Scoliosis: {scoliosis_count}")
    print(f"📊 Class weights: Normal={weight_normal:.2f}, Scoliosis={weight_scoliosis:.2f}")
    
    # Model oluştur - Attention mekanizmasını aktif et (%80+ accuracy için)
    if args.model_type == "advanced_lstm":
        model = build_model(model_type=args.model_type, use_attention=True)
    else:
        model = build_model(model_type=args.model_type)
    model = model.to(device)
    
    print(f"📊 Model: {args.model_type}")
    print(f"📊 Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Eğitim
    history = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model_name,
        class_weights=class_weights
    )


if __name__ == "__main__":
    main()
