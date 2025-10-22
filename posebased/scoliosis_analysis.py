#!/usr/bin/env python3
"""
Skolyoz Analizi için Derin Öğrenme Modeli
RGB video dosyalarından MediaPipe ile pose keypoint çıkarımı yapar ve 
MLP/LSTM modeli ile skolyoz/normal sınıflandırması yapar.

Girdi: RGB video dosyaları (MP4, AVI, vb.)
Çıktı: Her video için skolyoz/normal tahmini
"""

import os
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# MediaPipe pose detection setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class PoseExtractor:
    """MediaPipe kullanarak video karelerinden pose keypoint çıkarımı yapar"""
    
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Video dosyasından pose keypoint'leri çıkarır
        
        Args:
            video_path: Video dosyasının yolu
            
        Returns:
            keypoints: Shape (num_frames, num_keypoints*3) array
                      Her keypoint için (x, y, visibility) değerleri
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        if not cap.isOpened():
            print(f"Hata: Video dosyası açılamadı: {video_path}")
            return None
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR'den RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 33 keypoint'i çıkar (MediaPipe pose modeli)
                frame_keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                keypoints_list.append(frame_keypoints)
            else:
                # Eğer pose tespit edilemezse sıfırlarla doldur
                frame_keypoints = [0.0] * (33 * 3)  # 33 keypoint * 3 değer
                keypoints_list.append(frame_keypoints)
            
            frame_count += 1
            
            # İlerleme göster
            if frame_count % 30 == 0:
                print(f"İşlenen frame sayısı: {frame_count}")
        
        cap.release()
        
        if not keypoints_list:
            print(f"Uyarı: {video_path} dosyasından keypoint çıkarılamadı")
            return None
        
        keypoints_array = np.array(keypoints_list)
        print(f"Toplam {len(keypoints_list)} frame işlendi, keypoint shape: {keypoints_array.shape}")
        
        return keypoints_array


class ScoliosisDataset(Dataset):
    """Skolyoz analizi için PyTorch Dataset"""
    
    def __init__(self, keypoints_data: List[np.ndarray], labels: List[int], 
                 max_sequence_length: int = 100):
        """
        Args:
            keypoints_data: Her video için keypoint array'leri
            labels: Video etiketleri (0: Normal, 1: Skolyoz)
            max_sequence_length: Maksimum sequence uzunluğu
        """
        self.keypoints_data = keypoints_data
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        
        # Sequence'leri normalize et ve padding yap
        self.processed_data = self._preprocess_sequences()
    
    def _preprocess_sequences(self) -> List[np.ndarray]:
        """Sequence'leri normalize et ve padding yap"""
        processed = []
        
        for keypoints in self.keypoints_data:
            # Sequence uzunluğunu sınırla
            if len(keypoints) > self.max_sequence_length:
                keypoints = keypoints[:self.max_sequence_length]
            
            # Padding ekle
            if len(keypoints) < self.max_sequence_length:
                padding_length = self.max_sequence_length - len(keypoints)
                padding = np.zeros((padding_length, keypoints.shape[1]))
                keypoints = np.vstack([keypoints, padding])
            
            processed.append(keypoints)
        
        return processed
    
    def __len__(self):
        return len(self.keypoints_data)
    
    def __getitem__(self, idx):
        keypoints = torch.FloatTensor(self.processed_data[idx])
        label = torch.LongTensor([self.labels[idx]])
        return keypoints, label.squeeze()


class ScoliosisMLP(nn.Module):
    """Skolyoz analizi için MLP modeli"""
    
    def __init__(self, input_dim: int = 99, hidden_dims: List[int] = [256, 128, 64], 
                 num_classes: int = 2, dropout: float = 0.3):
        super(ScoliosisMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, features)
        # Global average pooling yaparak sequence dimension'ı kaldır
        x = torch.mean(x, dim=1)  # (batch_size, features)
        return self.network(x)


class ScoliosisLSTM(nn.Module):
    """Skolyoz analizi için LSTM modeli"""
    
    def __init__(self, input_dim: int = 99, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super(ScoliosisLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Son timestep'i al
        last_output = lstm_out[:, -1, :]
        
        # Dropout ve classification
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class ScoliosisAnalyzer:
    """Ana skolyoz analizi sınıfı"""
    
    def __init__(self, model_type: str = "lstm", device: str = "auto"):
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.scaler = StandardScaler()
        self.pose_extractor = PoseExtractor()
        
        print(f"Model tipi: {model_type}")
        print(f"Kullanılan cihaz: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """PyTorch device'ını belirle"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def prepare_dataset(self, video_paths: List[str], labels: List[int]) -> Tuple[DataLoader, DataLoader]:
        """
        Video dosyalarından dataset hazırla
        
        Args:
            video_paths: Video dosya yolları
            labels: Video etiketleri (0: Normal, 1: Skolyoz)
            
        Returns:
            train_loader, test_loader: PyTorch DataLoader'lar
        """
        print("Video dosyalarından keypoint çıkarımı yapılıyor...")
        
        all_keypoints = []
        valid_labels = []
        
        for i, (video_path, label) in enumerate(zip(video_paths, labels)):
            print(f"\n{i+1}/{len(video_paths)}: {video_path}")
            
            keypoints = self.pose_extractor.extract_keypoints_from_video(video_path)
            if keypoints is not None:
                all_keypoints.append(keypoints)
                valid_labels.append(label)
            else:
                print(f"Uyarı: {video_path} atlandı")
        
        print(f"\nToplam {len(all_keypoints)} video başarıyla işlendi")
        
        # Train-test split
        if len(all_keypoints) < 2:
            raise ValueError("En az 2 video gerekli!")
        
        X_train, X_test, y_train, y_test = train_test_split(
            all_keypoints, valid_labels, test_size=0.3, random_state=42, stratify=valid_labels
        )
        
        # Dataset'leri oluştur
        train_dataset = ScoliosisDataset(X_train, y_train)
        test_dataset = ScoliosisDataset(X_test, y_test)
        
        # DataLoader'ları oluştur
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        
        return train_loader, test_loader
    
    def create_model(self, input_dim: int = 99) -> nn.Module:
        """Model oluştur"""
        if self.model_type == "mlp":
            model = ScoliosisMLP(input_dim=input_dim)
        elif self.model_type == "lstm":
            model = ScoliosisLSTM(input_dim=input_dim)
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.model_type}")
        
        return model.to(self.device)
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, 
                   epochs: int = 50, learning_rate: float = 0.001):
        """
        Model eğitimi
        
        Args:
            train_loader: Eğitim verisi
            test_loader: Test verisi
            epochs: Eğitim epoch sayısı
            learning_rate: Öğrenme oranı
        """
        print(f"\n=== MODEL EĞİTİMİ BAŞLIYOR ===")
        print(f"Model tipi: {self.model_type}")
        print(f"Epoch sayısı: {epochs}")
        print(f"Öğrenme oranı: {learning_rate}")
        
        # Model oluştur
        input_dim = train_loader.dataset[0][0].shape[1]  # Feature dimension
        self.model = self.create_model(input_dim)
        
        # Optimizer ve loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Eğitim döngüsü
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Eğitim
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Test
            test_acc = self.evaluate_model(test_loader)
            test_accuracies.append(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        print(f"\n=== EĞİTİM TAMAMLANDI ===")
        print(f"Son test accuracy: {test_accuracies[-1]:.4f}")
        
        return train_losses, test_accuracies
    
    def evaluate_model(self, test_loader: DataLoader) -> float:
        """Model değerlendirmesi"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        return accuracy
    
    def predict_video(self, video_path: str) -> Dict[str, any]:
        """
        Tek video için tahmin yap
        
        Args:
            video_path: Video dosyası yolu
            
        Returns:
            prediction_dict: Tahmin sonuçları
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş! Önce train_model() çağırın.")
        
        print(f"\nTahmin yapılıyor: {video_path}")
        
        # Keypoint çıkarımı
        keypoints = self.pose_extractor.extract_keypoints_from_video(video_path)
        if keypoints is None:
            return {"error": "Video işlenemedi"}
        
        # Dataset formatına çevir
        dataset = ScoliosisDataset([keypoints], [0])  # Dummy label
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Tahmin
        self.model.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
        
        # Sonuçları hazırla
        result = {
            "video_path": video_path,
            "prediction": "Skolyoz" if prediction == 1 else "Normal",
            "confidence": {
                "Normal": probabilities[0][0].item(),
                "Skolyoz": probabilities[0][1].item()
            },
            "raw_probabilities": probabilities[0].cpu().numpy().tolist()
        }
        
        print(f"Tahmin: {result['prediction']}")
        print(f"Güven: Normal={result['confidence']['Normal']:.3f}, Skolyoz={result['confidence']['Skolyoz']:.3f}")
        
        return result
    
    def save_model(self, filepath: str):
        """Modeli kaydet"""
        if self.model is None:
            raise ValueError("Kaydedilecek model yok!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler
        }, filepath)
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath: str):
        """Modeli yükle"""
        # PyTorch 2.6+ uyumluluğu için weights_only=False kullan
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model_type = checkpoint['model_type']
        self.scaler = checkpoint['scaler']
        
        # Model oluştur ve ağırlıkları yükle
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model yüklendi: {filepath}")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Skolyoz Analizi")
    parser.add_argument("--videos", nargs="+", help="Video dosya yolları")
    parser.add_argument("--labels", nargs="+", type=int, help="Video etiketleri (0: Normal, 1: Skolyoz)")
    parser.add_argument("--model_type", choices=["mlp", "lstm"], default="lstm", help="Model tipi")
    parser.add_argument("--epochs", type=int, default=50, help="Eğitim epoch sayısı")
    parser.add_argument("--predict", help="Tahmin yapılacak video dosyası")
    parser.add_argument("--load_model", help="Yüklenecek model dosyası")
    parser.add_argument("--save_model", help="Model kaydetme dosyası")
    
    args = parser.parse_args()
    
    # Analyzer oluştur
    analyzer = ScoliosisAnalyzer(model_type=args.model_type)
    
    # Eğer model yüklenecekse
    if args.load_model:
        analyzer.load_model(args.load_model)
    
    # Eğitim modu
    if args.videos and args.labels:
        if len(args.videos) != len(args.labels):
            print("Hata: Video sayısı ile etiket sayısı eşleşmiyor!")
            return
        
        print(f"Eğitim için {len(args.videos)} video kullanılacak")
        
        # Dataset hazırla
        train_loader, test_loader = analyzer.prepare_dataset(args.videos, args.labels)
        
        # Model eğit
        train_losses, test_accuracies = analyzer.train_model(
            train_loader, test_loader, epochs=args.epochs
        )
        
        # Model kaydet
        if args.save_model:
            analyzer.save_model(args.save_model)
    
    # Tahmin modu
    elif args.predict:
        if analyzer.model is None:
            print("Hata: Tahmin için model gerekli! Önce eğitim yapın veya model yükleyin.")
            return
        
        result = analyzer.predict_video(args.predict)
        print(f"\n=== TAHMİN SONUCU ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        print("Kullanım:")
        print("Eğitim: python scoliosis_analysis.py --videos video1.mp4 video2.mp4 --labels 0 1")
        print("Tahmin: python scoliosis_analysis.py --predict video.mp4 --load_model model.pth")


if __name__ == "__main__":
    main()
