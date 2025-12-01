#!/usr/bin/env python3
"""
Yeni keypoint dosyalarını dataset'e ekle
"""

import os
import json
import numpy as np
import shutil
from pathlib import Path


def check_npy_format(npy_path):
    """NPY dosyasının formatını kontrol et"""
    data = np.load(npy_path)
    
    if len(data.shape) != 2:
        return False, f"Yanlış shape: {data.shape} (2D olmalı)"
    
    if data.shape[1] != 99:
        return False, f"Yanlış feature sayısı: {data.shape[1]} (99 olmalı: 33 keypoint x 3)"
    
    return True, f"OK - {data.shape[0]} frame, {data.shape[1]} feature"


def add_keypoints_to_dataset(source_folder, label, keypoints_dir="keypoints"):
    """
    Yeni keypoint dosyalarını dataset'e ekle
    
    Args:
        source_folder: Yeni NPY dosyalarının bulunduğu klasör
        label: "normal" veya "scoliosis"
        keypoints_dir: Hedef keypoints klasörü
    """
    if label not in ["normal", "scoliosis"]:
        print("❌ Label 'normal' veya 'scoliosis' olmalı!")
        return
    
    # Metadata yükle
    metadata_path = os.path.join(keypoints_dir, "metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = []
    
    # Mevcut dosya isimlerini al
    existing_names = {os.path.basename(m.get('keypoint_path', '')) for m in metadata}
    
    # Source klasördeki NPY dosyalarını bul
    source_path = Path(source_folder)
    npy_files = list(source_path.glob("*.npy"))
    
    if not npy_files:
        print(f"❌ {source_folder} klasöründe NPY dosyası bulunamadı!")
        return
    
    print(f"📁 {len(npy_files)} NPY dosyası bulundu")
    print(f"🏷️  Label: {label}")
    print()
    
    added_count = 0
    skipped_count = 0
    error_count = 0
    
    for npy_file in npy_files:
        filename = npy_file.name
        
        # Zaten var mı kontrol et
        if filename in existing_names:
            print(f"⏭️  {filename} - zaten mevcut, atlanıyor")
            skipped_count += 1
            continue
        
        # Format kontrol
        is_valid, msg = check_npy_format(str(npy_file))
        
        if not is_valid:
            print(f"❌ {filename} - {msg}")
            error_count += 1
            continue
        
        # Dosyayı kopyala
        dest_path = os.path.join(keypoints_dir, filename)
        shutil.copy2(str(npy_file), dest_path)
        
        # Metadata'ya ekle - front view olarak işaretle
        new_entry = {
            "video_path": f"dataset/{label}/front/{filename.replace('.npy', '.mp4')}",
            "keypoint_path": dest_path,
            "status": "success",
            "frame_count": int(np.load(dest_path).shape[0]),
            "added_manually": True
        }
        metadata.append(new_entry)
        
        print(f"✅ {filename} - {msg} - eklendi")
        added_count += 1
    
    # Metadata kaydet
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 50)
    print(f"📊 ÖZET:")
    print(f"  ✅ Eklenen: {added_count}")
    print(f"  ⏭️  Atlanan (mevcut): {skipped_count}")
    print(f"  ❌ Hatalı: {error_count}")
    print(f"  📁 Toplam metadata: {len(metadata)}")
    print("=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Yeni keypoint dosyalarını dataset'e ekle")
    parser.add_argument("--source", required=True, help="NPY dosyalarının bulunduğu klasör")
    parser.add_argument("--label", required=True, choices=["normal", "scoliosis"], 
                       help="Veri etiketi (normal veya scoliosis)")
    parser.add_argument("--keypoints_dir", default="keypoints", help="Hedef keypoints klasörü")
    
    args = parser.parse_args()
    
    add_keypoints_to_dataset(args.source, args.label, args.keypoints_dir)


if __name__ == "__main__":
    main()

