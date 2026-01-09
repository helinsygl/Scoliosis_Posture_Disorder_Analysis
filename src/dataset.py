#!/usr/bin/env python3
"""
Keypoint Dataset Loader
Keypoint dosyalarını yükler ve PyTorch Dataset formatına çevirir
"""

import os
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


class KeypointDataset(Dataset):
    """Keypoint verileri için PyTorch Dataset - İyileştirilmiş"""
    
    def __init__(self, keypoint_paths: List[str], labels: List[int], 
                 max_sequence_length: int = 150, normalize: bool = True,
                 augment: bool = False, is_training: bool = False):
        """
        Args:
            keypoint_paths: Keypoint dosya yolları
            labels: Etiketler (0: Normal, 1: Scoliosis)
            max_sequence_length: Maksimum sequence uzunluğu
            normalize: Keypoint'leri normalize et
            augment: Data augmentation kullan
            is_training: Training modunda mı (augmentation için)
        """
        self.keypoint_paths = keypoint_paths
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        self.normalize = normalize
        self.augment = augment and is_training
        self.is_training = is_training
        
        # Keypoint'leri yükle ve işle
        self.processed_data = self._load_and_process()
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """İyileştirilmiş normalizasyon - Z-score normalization"""
        normalized = keypoints.copy()
        
        # Her keypoint için ayrı ayrı normalize et
        for i in range(0, keypoints.shape[1], 3):
            # X koordinatı
            x_col = keypoints[:, i]
            if x_col.std() > 1e-8:
                normalized[:, i] = (x_col - x_col.mean()) / (x_col.std() + 1e-8)
            else:
                normalized[:, i] = x_col - x_col.mean()
            
            # Y koordinatı
            y_col = keypoints[:, i+1]
            if y_col.std() > 1e-8:
                normalized[:, i+1] = (y_col - y_col.mean()) / (y_col.std() + 1e-8)
            else:
                normalized[:, i+1] = y_col - y_col.mean()
            
            # Visibility değişmez
        
        return normalized
    
    def _augment_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Geliştirilmiş data augmentation - %80+ accuracy için"""
        augmented = keypoints.copy()
        
        # 1. Gaussian noise (daha geniş aralık)
        if np.random.rand() > 0.3:  # %70 şans
            noise_scale = np.random.uniform(0.01, 0.03)  # Daha geniş aralık
            for i in range(0, keypoints.shape[1], 3):
                augmented[:, i] += np.random.normal(0, noise_scale, keypoints.shape[0])
                augmented[:, i+1] += np.random.normal(0, noise_scale, keypoints.shape[0])
        
        # 2. Translation (daha geniş aralık)
        if np.random.rand() > 0.3:  # %70 şans
            tx = np.random.uniform(-0.08, 0.08)  # Daha geniş
            ty = np.random.uniform(-0.08, 0.08)
            for i in range(0, keypoints.shape[1], 3):
                augmented[:, i] += tx
                augmented[:, i+1] += ty
        
        # 3. Scaling (yeni - postür varyasyonları için)
        if np.random.rand() > 0.5:  # %50 şans
            scale = np.random.uniform(0.95, 1.05)  # Hafif ölçekleme
            # Merkez noktasını bul
            center_x = np.mean([augmented[:, i].mean() for i in range(0, keypoints.shape[1], 3)])
            center_y = np.mean([augmented[:, i+1].mean() for i in range(0, keypoints.shape[1], 3)])
            # Ölçekle
            for i in range(0, keypoints.shape[1], 3):
                augmented[:, i] = (augmented[:, i] - center_x) * scale + center_x
                augmented[:, i+1] = (augmented[:, i+1] - center_y) * scale + center_y
        
        # 4. Temporal subsampling (daha sık)
        if np.random.rand() > 0.5 and len(keypoints) > 20:  # %50 şans
            step = np.random.choice([2, 3])
            indices = np.arange(0, len(augmented), step)
            subsampled = augmented[indices]
            if len(subsampled) < len(augmented):
                ratio = len(augmented) / len(subsampled)
                new_indices = (np.arange(len(augmented)) / ratio).astype(int)
                new_indices = np.clip(new_indices, 0, len(subsampled) - 1)
                augmented = subsampled[new_indices]
        
        # 5. Keypoint dropout (yeni - eksik keypoint'lere karşı dayanıklılık)
        if np.random.rand() > 0.7:  # %30 şans
            dropout_rate = np.random.uniform(0.05, 0.15)  # %5-15 keypoint'i sıfırla
            num_keypoints = keypoints.shape[1] // 3
            num_to_drop = int(num_keypoints * dropout_rate)
            keypoint_indices = np.random.choice(num_keypoints, num_to_drop, replace=False)
            for idx in keypoint_indices:
                i = idx * 3
                augmented[:, i] = 0
                augmented[:, i+1] = 0
        
        return augmented
    
    def _load_and_process(self) -> List[np.ndarray]:
        """Keypoint'leri yükle ve işle"""
        processed = []
        
        for keypoint_path in self.keypoint_paths:
            if not os.path.exists(keypoint_path):
                print(f"Uyarı: Keypoint dosyası bulunamadı: {keypoint_path}")
                # Boş array oluştur
                keypoints = np.zeros((self.max_sequence_length, 99))
                processed.append(keypoints)
                continue
            
            # Keypoint'leri yükle
            keypoints = np.load(keypoint_path)
            
            # Normalize et (eğer istenirse) - Z-score kullan
            if self.normalize:
                keypoints = self._normalize_keypoints(keypoints)
            
            # Sequence uzunluğunu sınırla
            if len(keypoints) > self.max_sequence_length:
                # Training: rastgele bir başlangıç noktası seç
                if self.is_training:
                    start_idx = np.random.randint(0, len(keypoints) - self.max_sequence_length + 1)
                    keypoints = keypoints[start_idx:start_idx + self.max_sequence_length]
                else:
                    # Test: orta frame'leri al (daha stabil ve tutarlı)
                    start_idx = (len(keypoints) - self.max_sequence_length) // 2
                    keypoints = keypoints[start_idx:start_idx + self.max_sequence_length]
            
            # Padding ekle
            if len(keypoints) < self.max_sequence_length:
                padding_length = self.max_sequence_length - len(keypoints)
                padding = np.zeros((padding_length, keypoints.shape[1]))
                keypoints = np.vstack([keypoints, padding])
            
            processed.append(keypoints)
        
        return processed
    
    def __len__(self):
        return len(self.keypoint_paths)
    
    def __getitem__(self, idx):
        keypoints = self.processed_data[idx].copy()
        
        # Data augmentation (training sırasında)
        if self.augment:
            keypoints = self._augment_keypoints(keypoints)
        
        keypoints = torch.FloatTensor(keypoints)
        label = torch.LongTensor([self.labels[idx]])
        return keypoints, label.squeeze()


def extract_person_id(video_path: str) -> str:
    """
    Video path'inden kişi ID'sini çıkar
    HER VİDEO FARKLI BİR KİŞİ OLARAK KABUL EDİLİR (benzersiz ID)
    
    Bu, person-based split'in doğru çalışması için gereklidir.
    Eğer gerçekten aynı kişinin birden fazla videosu varsa,
    bu videolar aynı kişi ID'sine sahip olmalıdır.
    
    Örnekler:
        "kamera1_20251005_120022.avi" -> "kamera1_20251005_120022" (benzersiz)
        "kamera1_20251005_120132.avi" -> "kamera1_20251005_120132" (farklı kişi)
        "normal1.avi" -> "normal1" (benzersiz)
        "hasta25.mov" -> "hasta25" (benzersiz)
    """
    filename = os.path.basename(video_path)
    # Dosya adını (uzantı olmadan) kişi ID olarak kullan
    # Bu, her video'yu benzersiz bir kişi olarak kabul eder
    name_without_ext = filename.rsplit('.', 1)[0]
    return name_without_ext


def load_dataset_from_keypoints(keypoints_dir: str, test_size: float = 0.2, 
                                random_state: int = 42, augment: bool = True,
                                batch_size: int = 8, person_based_split: bool = True,
                                include_side_videos: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Keypoint klasöründen dataset yükle
    
    Args:
        keypoints_dir: Keypoint dosyalarının bulunduğu klasör
        test_size: Test set oranı
        random_state: Random seed
        include_side_videos: Side videoları da dahil et (default: False, sadece front)
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader'lar
    """
    # Metadata dosyasını yükle
    metadata_path = os.path.join(keypoints_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata dosyası bulunamadı: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Başarılı keypoint'leri filtrele
    successful = [m for m in metadata if m['status'] == 'success']
    
    # Video yollarından class, view ve person bilgisini çıkar
    keypoint_paths = []
    labels = []
    person_ids = []
    
    for item in successful:
        video_path = item['video_path']
        keypoint_path = item['keypoint_path']
        
        # Front/side yapısını kontrol et
        has_front_structure = '/front/' in video_path.lower() or '\\front\\' in video_path.lower()
        has_side_structure = '/side/' in video_path.lower() or '\\side\\' in video_path.lower()
        
        # Side videoları dahil etme kontrolü
        if not include_side_videos:
            # Sadece front videoları kullan (eğer front/side yapısı varsa)
            # Eğer side/ klasörü varsa ama front/ yoksa, side videoları atla
            if has_side_structure and not has_front_structure:
                continue  # Side videoları atla
        # include_side_videos=True ise tüm videoları kullan (front + side)
        
        # Class belirle (normal/scoliosis)
        if 'normal' in video_path.lower():
            label = 0
        elif 'scoliosis' in video_path.lower():
            label = 1
        else:
            continue  # Bilinmeyen class, atla
        
        # Person ID çıkar
        person_id = extract_person_id(video_path)
        
        keypoint_paths.append(keypoint_path)
        labels.append(label)
        person_ids.append(person_id)
    
    if len(keypoint_paths) < 2:
        raise ValueError("En az 2 keypoint dosyası gerekli!")
    
    # Person-based split (aynı kişi hem train hem test'te olmasın)
    if person_based_split:
        print("👥 Person-based split kullanılıyor (overfitting önleme)...")
        
        # Person'lara göre grupla
        person_to_indices = defaultdict(list)
        for idx, person_id in enumerate(person_ids):
            person_to_indices[person_id].append(idx)
        
        # Her person'ın tüm videolarını aynı sette tut
        unique_persons = list(person_to_indices.keys())
        person_labels = [labels[person_to_indices[pid][0]] for pid in unique_persons]
        
        # Person'ları class'a göre split et
        train_persons, test_persons = train_test_split(
            unique_persons, test_size=test_size,
            random_state=random_state, stratify=person_labels
        )
        
        # Person'lara göre index'leri topla
        train_indices = []
        test_indices = []
        
        for person_id in train_persons:
            train_indices.extend(person_to_indices[person_id])
        
        for person_id in test_persons:
            test_indices.extend(person_to_indices[person_id])
        
        X_train = [keypoint_paths[i] for i in train_indices]
        X_test = [keypoint_paths[i] for i in test_indices]
        y_train = [labels[i] for i in train_indices]
        y_test = [labels[i] for i in test_indices]
        
        print(f"  Toplam {len(unique_persons)} farklı kişi")
        print(f"  Train: {len(train_persons)} kişi, {len(X_train)} video")
        print(f"  Test: {len(test_persons)} kişi, {len(X_test)} video")
    else:
        # Normal random split
        X_train, X_test, y_train, y_test = train_test_split(
            keypoint_paths, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
    
    # Dataset'leri oluştur - augmentation ile
    train_dataset = KeypointDataset(X_train, y_train, augment=augment, is_training=True)
    test_dataset = KeypointDataset(X_test, y_test, augment=False, is_training=False)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    view_type = "FRONT + SIDE VİDEOLAR" if include_side_videos else "SADECE FRONT VİDEOLAR"
    print(f"📊 Dataset yüklendi ({view_type}):")
    print(f"  Train: {len(X_train)} örnek (Normal: {y_train.count(0)}, Scoliosis: {y_train.count(1)})")
    print(f"  Test: {len(X_test)} örnek (Normal: {y_test.count(0)}, Scoliosis: {y_test.count(1)})")
    print(f"  Toplam: {len(labels)} örnek (Normal: {labels.count(0)}, Scoliosis: {labels.count(1)})")
    print(f"  Person-based split: {'Aktif ✅' if person_based_split else 'Kapalı ❌'}")
    print(f"  Augmentation: {'Aktif ✅' if augment else 'Kapalı ❌'}")
    print(f"  Side videolar: {'Dahil ✅' if include_side_videos else 'Hariç ❌'}")
    
    return train_loader, test_loader
