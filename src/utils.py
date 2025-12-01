#!/usr/bin/env python3
"""
Yardımcı fonksiyonlar
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def save_checkpoint(model, optimizer, epoch, val_acc, filepath):
    """Model checkpoint kaydet"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint kaydedildi: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Model checkpoint yükle"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    
    print(f"📂 Checkpoint yüklendi: {filepath}")
    print(f"  Epoch: {epoch}, Val Acc: {val_acc:.2f}%")
    
    return epoch, val_acc


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def calculate_metrics(y_true, y_pred, labels=['Normal', 'Scoliosis']):
    """Metrikleri hesapla"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    metrics['per_class'] = {}
    for i, label in enumerate(labels):
        metrics['per_class'][label] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i]
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics


def save_metrics(metrics, filepath):
    """Metrikleri JSON olarak kaydet"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Metrikler kaydedildi: {filepath}")


def print_metrics(metrics):
    """Metrikleri yazdır"""
    print("\n" + "="*50)
    print("📊 DEĞERLENDİRME METRİKLERİ")
    print("="*50)
    
    print(f"\n🎯 Genel Metrikler:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n📈 Sınıf Bazlı Metrikler:")
    for label, class_metrics in metrics['per_class'].items():
        print(f"  {label}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")
    
    print(f"\n🔢 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"              Predicted")
    print(f"              Normal  Scoliosis")
    print(f"  Actual Normal    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"         Scoliosis {cm[1][0]:4d}      {cm[1][1]:4d}")
    
    print("="*50 + "\n")
