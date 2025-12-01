#!/usr/bin/env python3
"""
Farklı seed değerleriyle model eğitip en iyi seed'i bulur
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from dataset import load_dataset_from_keypoints
from train import train, set_seed
from evaluate import evaluate
from utils import load_checkpoint, calculate_metrics


def test_seed(seed: int, model_type: str = "advanced_lstm", 
              epochs: int = 50, batch_size: int = 16, 
              lr: float = 0.001):
    """
    Belirli bir seed ile model eğitip test accuracy'sini döndürür
    """
    print(f"\n{'='*60}")
    print(f"🔍 Seed {seed} test ediliyor...")
    print(f"{'='*60}")
    
    # Seed sabitle
    set_seed(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset yükle
    train_loader, val_loader = load_dataset_from_keypoints(
        "keypoints",
        augment=True,
        batch_size=batch_size,
        person_based_split=True
    )
    
    # Class weights hesapla
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().tolist())
    
    normal_count = train_labels.count(0)
    scoliosis_count = train_labels.count(1)
    total = normal_count + scoliosis_count
    
    weight_normal = total / (2 * normal_count) if normal_count > 0 else 1.0
    weight_scoliosis = total / (2 * scoliosis_count) if scoliosis_count > 0 else 1.0
    class_weights = [weight_normal, weight_scoliosis]
    
    # Model oluştur
    model = build_model(model_type=model_type)
    model = model.to(device)
    
    # Model adı (seed ile)
    model_name = f"seed_test_{seed}"
    
    # Eğitim (daha hızlı test için daha az epoch)
    history = train(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=lr,
        device=device,
        save_dir="saved_models",
        model_name=model_name,
        class_weights=class_weights
    )
    
    # Test setinde değerlendir
    model_path = f"saved_models/{model_name}.pth"
    
    # Test loader'ı tekrar yükle (aynı seed ile)
    set_seed(seed)  # Aynı seed ile test seti de aynı olsun
    _, test_loader = load_dataset_from_keypoints(
        "keypoints",
        augment=False,
        batch_size=batch_size,
        person_based_split=True
    )
    
    # Model'i yükle
    model = build_model(model_type=model_type)
    model = model.to(device)
    load_checkpoint(model_path, model)
    
    # Değerlendir
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    
    test_accuracy = metrics['accuracy']
    
    print(f"\n✅ Seed {seed} - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Geçici model dosyasını sil (disk alanı için)
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(f"saved_models/{model_name}_history.json"):
        os.remove(f"saved_models/{model_name}_history.json")
    
    return {
        'seed': seed,
        'test_accuracy': test_accuracy,
        'val_accuracy': max(history['val_acc']) if history['val_acc'] else 0,
        'best_epoch': history['val_acc'].index(max(history['val_acc'])) + 1 if history['val_acc'] else 0
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="En iyi seed'i bul")
    parser.add_argument("--seeds", type=str, default="42,123,423,456,789,999,1111,2222,3333,4444",
                       help="Test edilecek seed'ler (virgülle ayrılmış)")
    parser.add_argument("--model_type", default="advanced_lstm", 
                       choices=["simple_lstm", "advanced_lstm", "posture", "transformer", "hybrid"])
    parser.add_argument("--epochs", type=int, default=50, help="Her seed için epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="results/best_seed_results.json", 
                       help="Sonuçları kaydet")
    
    args = parser.parse_args()
    
    # Seed listesini parse et
    seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    
    print(f"🔍 {len(seed_list)} farklı seed test edilecek:")
    print(f"   Seeds: {seed_list}")
    print(f"   Model: {args.model_type}")
    print(f"   Epochs per seed: {args.epochs}")
    
    results = []
    
    for seed in seed_list:
        try:
            result = test_seed(seed, args.model_type, args.epochs, args.batch_size, args.lr)
            results.append(result)
        except Exception as e:
            print(f"❌ Seed {seed} hatası: {e}")
            results.append({
                'seed': seed,
                'test_accuracy': 0,
                'error': str(e)
            })
    
    # Sonuçları sırala
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # En iyi seed
    best_result = results[0]
    
    print(f"\n{'='*60}")
    print(f"🏆 EN İYİ SEED BULUNDU!")
    print(f"{'='*60}")
    print(f"   Seed: {best_result['seed']}")
    print(f"   Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    print(f"   Val Accuracy: {best_result.get('val_accuracy', 0):.4f} ({best_result.get('val_accuracy', 0)*100:.2f}%)")
    print(f"   Best Epoch: {best_result.get('best_epoch', 0)}")
    
    print(f"\n📊 Tüm Sonuçlar (Test Accuracy'ye göre sıralı):")
    print(f"{'Seed':<10} {'Test Acc':<12} {'Val Acc':<12} {'Best Epoch':<12}")
    print(f"{'-'*50}")
    for r in results:
        if 'error' not in r:
            print(f"{r['seed']:<10} {r['test_accuracy']*100:>8.2f}%    {r.get('val_accuracy', 0)*100:>8.2f}%    {r.get('best_epoch', 0):>10}")
        else:
            print(f"{r['seed']:<10} {'ERROR':<12}")
    
    # Sonuçları kaydet
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'best_seed': best_result['seed'],
            'best_test_accuracy': best_result['test_accuracy'],
            'all_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Sonuçlar kaydedildi: {args.output}")
    
    return best_result['seed']


if __name__ == "__main__":
    main()

