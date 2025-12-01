#!/usr/bin/env python3
"""
Model Değerlendirme Scripti
Test seti üzerinde metrikler hesaplar
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from model import build_model
from dataset import load_dataset_from_keypoints
from utils import load_checkpoint, calculate_metrics, save_metrics, print_metrics


def evaluate(model, test_loader, device):
    """Model değerlendirmesi"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Model Değerlendirmesi")
    parser.add_argument("--keypoints_dir", default="keypoints", help="Keypoint klasörü")
    parser.add_argument("--model_path", required=True, help="Model checkpoint yolu")
    parser.add_argument("--model_type", default="simple_lstm",
                       choices=["simple_lstm", "advanced_lstm", "posture", "transformer", "hybrid"],
                       help="Model tipi")
    parser.add_argument("--output_dir", default="results", help="Sonuç kayıt klasörü")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Dataset yükle - augmentation kapalı, person-based split aktif
    _, test_loader = load_dataset_from_keypoints(
        args.keypoints_dir, 
        augment=False,
        batch_size=16,
        person_based_split=True  # Person-based split (overfitting önleme)
    )
    
    # Model oluştur ve yükle
    model = build_model(model_type=args.model_type)
    model = model.to(device)
    
    load_checkpoint(args.model_path, model)
    
    # Değerlendirme
    print("🔮 Model değerlendirmesi başlıyor...")
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(y_true, y_pred)
    
    # Metrikleri yazdır
    print_metrics(metrics)
    
    # Metrikleri kaydet
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    save_metrics(metrics, metrics_path)
    
    # Detaylı sonuçları kaydet
    results = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_probs': y_probs.tolist(),
        'metrics': metrics
    }
    
    results_path = os.path.join(args.output_dir, "detailed_results.json")
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Sonuçlar kaydedildi:")
    print(f"  Metrikler: {metrics_path}")
    print(f"  Detaylı sonuçlar: {results_path}")


if __name__ == "__main__":
    main()
