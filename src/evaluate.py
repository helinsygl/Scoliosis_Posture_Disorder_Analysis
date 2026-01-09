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


def evaluate(model, test_loader, device, test_dataset=None):
    """Model değerlendirmesi"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    video_paths = []
    
    with torch.no_grad():
        batch_idx = 0
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Video path'lerini al
            if test_dataset is not None:
                batch_size = len(target)
                for i in range(batch_size):
                    idx = batch_idx * test_loader.batch_size + i
                    if idx < len(test_dataset.keypoint_paths):
                        # Keypoint path'inden video path'i bul
                        keypoint_path = test_dataset.keypoint_paths[idx]
                        video_paths.append(keypoint_path)  # Şimdilik keypoint path, sonra video path'e çevrilecek
            
            batch_idx += 1
    
    return np.array(all_targets), np.array(all_preds), np.array(all_probs), video_paths


def main():
    parser = argparse.ArgumentParser(description="Model Değerlendirmesi")
    parser.add_argument("--keypoints_dir", default="keypoints", help="Keypoint klasörü")
    parser.add_argument("--model_path", required=True, help="Model checkpoint yolu")
    parser.add_argument("--model_type", default="simple_lstm",
                       choices=["simple_lstm", "advanced_lstm", "posture", "transformer", "hybrid"],
                       help="Model tipi")
    parser.add_argument("--output_dir", default="results", help="Sonuç kayıt klasörü")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--include_side", action="store_true", help="Side videoları da dahil et (front + side)")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Dataset yükle - augmentation kapalı, person-based split aktif
    # Test dataset'i de almak için dataset.py'yi modifiye etmemiz gerekiyor
    # Şimdilik metadata'dan video path'lerini alacağız
    import json
    from dataset import KeypointDataset
    from sklearn.model_selection import train_test_split
    from collections import defaultdict
    
    # Metadata yükle
    metadata_path = os.path.join(args.keypoints_dir, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Test setindeki videoları bul
    successful = [m for m in metadata if m['status'] == 'success']
    keypoint_paths = []
    labels = []
    person_ids = []
    video_paths_list = []
    
    for item in successful:
        video_path = item['video_path']
        keypoint_path = item['keypoint_path']
        
        has_front_structure = '/front/' in video_path.lower() or '\\front\\' in video_path.lower()
        has_side_structure = '/side/' in video_path.lower() or '\\side\\' in video_path.lower()
        
        # Side videoları dahil etme kontrolü
        if not args.include_side:
            # Sadece front videoları kullan
            if has_side_structure and not has_front_structure:
                continue
        # include_side=True ise tüm videoları kullan (front + side)
        
        if 'normal' in video_path.lower():
            label = 0
        elif 'scoliosis' in video_path.lower():
            label = 1
        else:
            continue
        
        from dataset import extract_person_id
        person_id = extract_person_id(video_path)
        
        keypoint_paths.append(keypoint_path)
        labels.append(label)
        person_ids.append(person_id)
        video_paths_list.append(video_path)
    
    # Person-based split (evaluate.py ile aynı seed kullan)
    person_to_indices = defaultdict(list)
    for idx, person_id in enumerate(person_ids):
        person_to_indices[person_id].append(idx)
    
    unique_persons = list(person_to_indices.keys())
    person_labels = [labels[person_to_indices[pid][0]] for pid in unique_persons]
    
    train_persons, test_persons = train_test_split(
        unique_persons, test_size=0.2,
        random_state=42, stratify=person_labels
    )
    
    test_indices = []
    for person_id in test_persons:
        test_indices.extend(person_to_indices[person_id])
    
    X_test = [keypoint_paths[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    test_video_paths = [video_paths_list[i] for i in test_indices]
    
    # Test dataset oluştur
    test_dataset = KeypointDataset(X_test, y_test, augment=False, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Model oluştur ve yükle - Attention mekanizmasını aktif et (eğer advanced_lstm ise)
    if args.model_type == "advanced_lstm":
        model = build_model(model_type=args.model_type, use_attention=True)
    else:
        model = build_model(model_type=args.model_type)
    model = model.to(device)
    
    load_checkpoint(args.model_path, model)
    
    # Değerlendirme
    print("🔮 Model değerlendirmesi başlıyor...")
    y_true, y_pred, y_probs, video_paths = evaluate(model, test_loader, device, test_dataset)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(y_true, y_pred)
    
    # Metrikleri yazdır
    print_metrics(metrics)
    
    # Metrikleri kaydet
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    save_metrics(metrics, metrics_path)
    
    # Metadata'dan keypoint path'e göre video path'leri bul
    keypoint_to_video = {}
    for item in metadata:
        if item['status'] == 'success':
            keypoint_to_video[item['keypoint_path']] = item['video_path']
    
    # Video isimlerini ve sonuçları hazırla
    label_names = ['Normal', 'Scoliosis']
    video_results = []
    for i, keypoint_path in enumerate(video_paths):
        # Keypoint path'inden video path'i bul
        actual_video_path = keypoint_to_video.get(keypoint_path, keypoint_path)
        video_name = os.path.basename(actual_video_path)
        true_label = label_names[y_true[i]]
        pred_label = label_names[y_pred[i]]
        is_correct = y_true[i] == y_pred[i]
        
        video_results.append({
            'video_name': video_name,
            'video_path': actual_video_path,
            'true_label': true_label,
            'predicted_label': pred_label,
            'is_correct': bool(is_correct),
            'confidence': {
                'Normal': float(y_probs[i][0]),
                'Scoliosis': float(y_probs[i][1])
            }
        })
    
    # Detaylı sonuçları kaydet
    results = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_probs': y_probs.tolist(),
        'video_results': video_results,
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
