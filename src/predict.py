#!/usr/bin/env python3
"""
Tek video için tahmin scripti
Eğitilmiş model ile yeni video üzerinde skolyoz/normal tahmini yapar
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from pathlib import Path

from model import build_model
from extract_keypoints import PoseExtractor
from dataset import KeypointDataset
from torch.utils.data import DataLoader
from utils import load_checkpoint


def predict_video(model, video_path: str, extractor: PoseExtractor, 
                  device: str = "cuda", max_sequence_length: int = 100):
    """
    Tek video için tahmin yap
    
    Args:
        model: Eğitilmiş model
        video_path: Video dosyası yolu
        extractor: Pose keypoint extractor
        device: Device (cuda/cpu)
        max_sequence_length: Maksimum sequence uzunluğu
        
    Returns:
        prediction_dict: Tahmin sonuçları
    """
    print(f"\n🎬 Video işleniyor: {video_path}")
    
    # Keypoint çıkarımı
    keypoints = extractor.extract_keypoints_from_video(video_path)
    
    if keypoints is None:
        return {
            "error": "Video işlenemedi - pose tespit edilemedi",
            "video_path": video_path
        }
    
    # Normalize et
    for i in range(0, keypoints.shape[1], 3):
        if keypoints[:, i].max() > 0:
            keypoints[:, i] = (keypoints[:, i] - keypoints[:, i].min()) / \
                             (keypoints[:, i].max() - keypoints[:, i].min() + 1e-8)
        if keypoints[:, i+1].max() > 0:
            keypoints[:, i+1] = (keypoints[:, i+1] - keypoints[:, i+1].min()) / \
                              (keypoints[:, i+1].max() - keypoints[:, i+1].min() + 1e-8)
    
    # Sequence uzunluğunu sınırla
    if len(keypoints) > max_sequence_length:
        keypoints = keypoints[:max_sequence_length]
    
    # Padding ekle
    if len(keypoints) < max_sequence_length:
        padding_length = max_sequence_length - len(keypoints)
        padding = np.zeros((padding_length, keypoints.shape[1]))
        keypoints = np.vstack([keypoints, padding])
    
    # Tensor'a çevir
    keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
    
    # Tahmin
    model.eval()
    with torch.no_grad():
        output = model(keypoints_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
    
    # Sonuçları hazırla
    result = {
        "video_path": video_path,
        "prediction": "Skolyoz" if prediction == 1 else "Normal",
        "prediction_class": prediction,
        "confidence": {
            "Normal": float(probabilities[0][0].item()),
            "Skolyoz": float(probabilities[0][1].item())
        },
        "raw_probabilities": probabilities[0].cpu().numpy().tolist(),
        "num_frames": len(keypoints)
    }
    
    return result


def predict_batch(model, video_paths: list, extractor: PoseExtractor,
                  device: str = "cuda", output_file: str = None):
    """
    Toplu video tahmini
    
    Args:
        model: Eğitilmiş model
        video_paths: Video dosya yolları listesi
        extractor: Pose keypoint extractor
        device: Device (cuda/cpu)
        output_file: Sonuçları kaydetmek için dosya yolu (opsiyonel)
    """
    results = []
    
    print(f"🔮 Toplu tahmin başlıyor: {len(video_paths)} video")
    
    for i, video_path in enumerate(video_paths):
        print(f"\n[{i+1}/{len(video_paths)}] İşleniyor: {os.path.basename(video_path)}")
        
        try:
            result = predict_video(model, video_path, extractor, device)
            results.append(result)
            
            if "error" not in result:
                print(f"  ✅ Tahmin: {result['prediction']}")
                print(f"  📊 Güven: Normal={result['confidence']['Normal']:.3f}, "
                      f"Skolyoz={result['confidence']['Skolyoz']:.3f}")
            else:
                print(f"  ❌ Hata: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ Hata: {e}")
            results.append({
                "video_path": video_path,
                "error": str(e)
            })
    
    # Sonuçları kaydet
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Sonuçlar kaydedildi: {output_file}")
    
    # Özet istatistikler
    successful = [r for r in results if "error" not in r]
    if successful:
        normal_count = sum(1 for r in successful if r['prediction'] == 'Normal')
        scoliosis_count = sum(1 for r in successful if r['prediction'] == 'Skolyoz')
        
        print(f"\n📊 Özet:")
        print(f"  Başarılı tahmin: {len(successful)}/{len(video_paths)}")
        print(f"  Normal: {normal_count}")
        print(f"  Skolyoz: {scoliosis_count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Video Tahmin Scripti")
    parser.add_argument("--model_path", required=True, help="Eğitilmiş model checkpoint yolu")
    parser.add_argument("--model_type", default="advanced_lstm",
                       choices=["advanced_lstm", "transformer", "hybrid"],
                       help="Model tipi")
    parser.add_argument("--video", help="Tek video dosyası yolu")
    parser.add_argument("--video_dir", help="Video klasörü (toplu tahmin için)")
    parser.add_argument("--output", help="Sonuç kayıt dosyası (JSON)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Model oluştur ve yükle
    print(f"📂 Model yükleniyor: {args.model_path}")
    model = build_model(model_type=args.model_type)
    model = model.to(device)
    
    load_checkpoint(args.model_path, model)
    
    # Pose extractor
    extractor = PoseExtractor()
    
    # Tahmin
    if args.video:
        # Tek video tahmini
        result = predict_video(model, args.video, extractor, device)
        
        print(f"\n{'='*50}")
        print(f"📊 TAHMİN SONUCU")
        print(f"{'='*50}")
        print(f"Video: {result['video_path']}")
        if "error" not in result:
            print(f"Tahmin: {result['prediction']}")
            print(f"Güven Skorları:")
            print(f"  Normal:   {result['confidence']['Normal']:.4f} ({result['confidence']['Normal']*100:.2f}%)")
            print(f"  Skolyoz:  {result['confidence']['Skolyoz']:.4f} ({result['confidence']['Skolyoz']*100:.2f}%)")
            print(f"Frame sayısı: {result['num_frames']}")
        else:
            print(f"❌ Hata: {result['error']}")
        print(f"{'='*50}\n")
        
        # Sonuçları kaydet
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Sonuç kaydedildi: {args.output}")
    
    elif args.video_dir:
        # Toplu tahmin
        import glob
        video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.wmv', '*.flv']
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(args.video_dir, ext)))
            video_paths.extend(glob.glob(os.path.join(args.video_dir, '**', ext), recursive=True))
        
        if not video_paths:
            print(f"❌ {args.video_dir} klasöründe video bulunamadı!")
            return
        
        print(f"📹 {len(video_paths)} video bulundu")
        
        results = predict_batch(model, video_paths, extractor, device, args.output)
    
    else:
        print("❌ --video veya --video_dir parametresi gerekli!")
        parser.print_help()


if __name__ == "__main__":
    main()
