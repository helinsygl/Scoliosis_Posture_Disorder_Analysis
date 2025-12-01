#!/usr/bin/env python3
"""
Gelişmiş Skolyoz Analizi Modelleri
LSTM, Transformer ve hibrit modeller
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class AdvancedLSTM(nn.Module):
    """Basitleştirilmiş LSTM modeli - Küçük dataset için optimize edilmiş"""
    
    def __init__(self, input_dim: int = 99, hidden_dim: int = 64, 
                 num_layers: int = 1, num_classes: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = True,
                 use_attention: bool = False):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Basit LSTM - küçük ve hızlı
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0,
            bidirectional=bidirectional
        )
        
        # Simple attention (opsiyonel)
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        if use_attention:
            self.attention_weights = nn.Linear(lstm_output_dim, 1)
        
        # Basit classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # Input normalization (batch_first için transpose gerekli)
        batch_size, seq_len, features = x.shape
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        if self.use_attention:
            # Simple attention
            attn_weights = torch.softmax(self.attention_weights(lstm_out), dim=1)
            output = torch.sum(lstm_out * attn_weights, dim=1)
        else:
            # Global average pooling (en basit ve etkili)
            output = torch.mean(lstm_out, dim=1)
        
        # Classification
        output = self.dropout(output)
        output = self.fc(output)
        
        return output


class SimpleLSTM(nn.Module):
    """En basit LSTM - küçük dataset için ideal"""
    
    def __init__(self, input_dim: int = 99, hidden_dim: int = 32, 
                 num_classes: int = 2, dropout: float = 0.2):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Global average pooling
        output = torch.mean(lstm_out, dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return output


class PostureFeatureModel(nn.Module):
    """Özel postür özellikleri çıkaran model - Skolyoz için optimize edilmiş"""
    
    def __init__(self, input_dim: int = 99, num_classes: int = 2, dropout: float = 0.3):
        super(PostureFeatureModel, self).__init__()
        
        # Keypoint indices (MediaPipe Pose)
        # 11: left_shoulder, 12: right_shoulder
        # 23: left_hip, 24: right_hip
        # 0: nose (head reference)
        
        # Feature extraction
        self.feature_fc = nn.Sequential(
            nn.Linear(15, 32),  # 15 özel postür özelliği
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        
        # Temporal analysis with small LSTM
        self.lstm = nn.LSTM(16, 32, num_layers=1, batch_first=True, bidirectional=True)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def extract_posture_features(self, keypoints):
        """
        Keypoints'ten postür özelliklerini çıkar
        keypoints: (batch, seq_len, 99) - 33 keypoint * 3 (x, y, visibility)
        """
        batch_size, seq_len, _ = keypoints.shape
        
        # Keypoint coordinates (her 3 değer: x, y, visibility)
        # İndeksler: keypoint_idx * 3 + {0: x, 1: y, 2: vis}
        
        # Shoulder keypoints
        left_shoulder_x = keypoints[:, :, 11*3]
        left_shoulder_y = keypoints[:, :, 11*3 + 1]
        right_shoulder_x = keypoints[:, :, 12*3]
        right_shoulder_y = keypoints[:, :, 12*3 + 1]
        
        # Hip keypoints
        left_hip_x = keypoints[:, :, 23*3]
        left_hip_y = keypoints[:, :, 23*3 + 1]
        right_hip_x = keypoints[:, :, 24*3]
        right_hip_y = keypoints[:, :, 24*3 + 1]
        
        # Nose (head reference)
        nose_x = keypoints[:, :, 0*3]
        nose_y = keypoints[:, :, 0*3 + 1]
        
        # === POSTURE FEATURES ===
        
        # 1. Shoulder tilt (omuz eğikliği) - skolyozda asimetrik
        shoulder_tilt = right_shoulder_y - left_shoulder_y
        
        # 2. Hip tilt (kalça eğikliği)
        hip_tilt = right_hip_y - left_hip_y
        
        # 3. Shoulder-hip alignment difference
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
        hip_mid_x = (left_hip_x + right_hip_x) / 2
        spine_deviation = shoulder_mid_x - hip_mid_x
        
        # 4. Head alignment relative to shoulders
        head_deviation = nose_x - shoulder_mid_x
        
        # 5. Shoulder width ratio
        shoulder_width = torch.abs(right_shoulder_x - left_shoulder_x)
        
        # 6. Hip width ratio
        hip_width = torch.abs(right_hip_x - left_hip_x)
        
        # 7. Torso symmetry (left vs right distance)
        left_torso = torch.sqrt((left_shoulder_x - left_hip_x)**2 + (left_shoulder_y - left_hip_y)**2)
        right_torso = torch.sqrt((right_shoulder_x - right_hip_x)**2 + (right_shoulder_y - right_hip_y)**2)
        torso_asymmetry = left_torso - right_torso
        
        # 8. Absolute values for severity
        abs_shoulder_tilt = torch.abs(shoulder_tilt)
        abs_hip_tilt = torch.abs(hip_tilt)
        abs_spine_deviation = torch.abs(spine_deviation)
        
        # 9. Tilt ratio
        tilt_ratio = shoulder_tilt / (hip_tilt + 1e-6)
        
        # 10. Cross-body diagonal differences
        diag1 = torch.sqrt((left_shoulder_x - right_hip_x)**2 + (left_shoulder_y - right_hip_y)**2)
        diag2 = torch.sqrt((right_shoulder_x - left_hip_x)**2 + (right_shoulder_y - left_hip_y)**2)
        diagonal_diff = diag1 - diag2
        
        # Stack all features
        features = torch.stack([
            shoulder_tilt, hip_tilt, spine_deviation, head_deviation,
            shoulder_width, hip_width, torso_asymmetry,
            abs_shoulder_tilt, abs_hip_tilt, abs_spine_deviation,
            tilt_ratio, diagonal_diff, left_torso, right_torso,
            (diag1 + diag2) / 2  # Average diagonal
        ], dim=-1)  # (batch, seq_len, 15)
        
        return features
    
    def forward(self, x):
        # Extract posture-specific features
        features = self.extract_posture_features(x)
        
        # Feature transformation
        features = self.feature_fc(features)
        
        # Temporal analysis
        lstm_out, _ = self.lstm(features)
        
        # Global pooling
        output = torch.mean(lstm_out, dim=1)
        
        # Classification
        output = self.classifier(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer tabanlı model"""
    
    def __init__(self, input_dim: int = 99, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 4,
                 num_classes: int = 2, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridModel(nn.Module):
    """LSTM + Transformer hibrit model"""
    
    def __init__(self, input_dim: int = 99, lstm_hidden: int = 128,
                 d_model: int = 256, nhead: int = 8, num_transformer_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3):
        super(HybridModel, self).__init__()
        
        # LSTM feature extractor
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=lstm_hidden * 2,
            nhead=nhead,
            dim_feedforward=(lstm_hidden * 2) * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_transformer_layers)
        
        # Classification head
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # LSTM feature extraction
        lstm_out, _ = self.lstm(x)
        
        # Transformer encoding
        transformer_out = self.transformer(lstm_out)
        
        # Global average pooling
        output = torch.mean(transformer_out, dim=1)
        
        # Classification
        output = self.fc1(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


def build_model(model_type: str = "simple_lstm", **kwargs) -> nn.Module:
    """
    Model oluştur
    
    Args:
        model_type: Model tipi ('simple_lstm', 'advanced_lstm', 'posture', 'transformer', 'hybrid')
        **kwargs: Model parametreleri
        
    Returns:
        Model instance
    """
    if model_type == "simple_lstm":
        return SimpleLSTM(**kwargs)
    elif model_type == "advanced_lstm":
        return AdvancedLSTM(**kwargs)
    elif model_type == "posture":
        return PostureFeatureModel(**kwargs)
    elif model_type == "transformer":
        return TransformerModel(**kwargs)
    elif model_type == "hybrid":
        return HybridModel(**kwargs)
    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")
