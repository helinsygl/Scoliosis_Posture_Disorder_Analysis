#!/usr/bin/env python3
"""
Klasik ML ile Skolyoz Analizi
Random Forest, SVM, XGBoost - küçük datasetler için daha iyi performans
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def extract_posture_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Keypoint serisinden postür özelliklerini çıkar
    Her frame için özellikler çıkarıp istatistiksel özet al
    
    Args:
        keypoints: (seq_len, 99) boyutunda keypoint dizisi
        
    Returns:
        Özellik vektörü
    """
    features = []
    
    # Her frame için özellik çıkar
    for frame_idx in range(len(keypoints)):
        kp = keypoints[frame_idx]
        
        # Keypoint coordinates (her 3 değer: x, y, visibility)
        def get_point(idx):
            return kp[idx*3], kp[idx*3 + 1], kp[idx*3 + 2]
        
        # Shoulder keypoints (11: left, 12: right)
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        
        # Hip keypoints (23: left, 24: right)
        left_hip = get_point(23)
        right_hip = get_point(24)
        
        # Nose (0)
        nose = get_point(0)
        
        # Ear keypoints (7: left, 8: right)
        left_ear = get_point(7)
        right_ear = get_point(8)
        
        # === POSTURE FEATURES ===
        frame_features = []
        
        # 1. Shoulder tilt (omuz eğikliği)
        shoulder_tilt = right_shoulder[1] - left_shoulder[1]
        frame_features.append(shoulder_tilt)
        
        # 2. Hip tilt (kalça eğikliği)
        hip_tilt = right_hip[1] - left_hip[1]
        frame_features.append(hip_tilt)
        
        # 3. Shoulder-hip alignment difference (omurga eğriliği)
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        spine_deviation = shoulder_mid_x - hip_mid_x
        frame_features.append(spine_deviation)
        
        # 4. Head alignment
        head_deviation = nose[0] - shoulder_mid_x
        frame_features.append(head_deviation)
        
        # 5. Shoulder width
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        frame_features.append(shoulder_width)
        
        # 6. Hip width
        hip_width = abs(right_hip[0] - left_hip[0])
        frame_features.append(hip_width)
        
        # 7. Torso asymmetry
        left_torso = np.sqrt((left_shoulder[0] - left_hip[0])**2 + (left_shoulder[1] - left_hip[1])**2)
        right_torso = np.sqrt((right_shoulder[0] - right_hip[0])**2 + (right_shoulder[1] - right_hip[1])**2)
        torso_asymmetry = left_torso - right_torso
        frame_features.append(torso_asymmetry)
        
        # 8. Diagonal differences
        diag1 = np.sqrt((left_shoulder[0] - right_hip[0])**2 + (left_shoulder[1] - right_hip[1])**2)
        diag2 = np.sqrt((right_shoulder[0] - left_hip[0])**2 + (right_shoulder[1] - left_hip[1])**2)
        diagonal_diff = diag1 - diag2
        frame_features.append(diagonal_diff)
        
        # 9. Ear tilt (baş eğikliği)
        ear_tilt = right_ear[1] - left_ear[1]
        frame_features.append(ear_tilt)
        
        # 10. Absolute values
        frame_features.append(abs(shoulder_tilt))
        frame_features.append(abs(hip_tilt))
        frame_features.append(abs(spine_deviation))
        
        features.append(frame_features)
    
    features = np.array(features)
    
    # İstatistiksel özet çıkar (mean, std, min, max, median)
    stats = []
    for i in range(features.shape[1]):
        col = features[:, i]
        stats.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            np.median(col),
            np.percentile(col, 25),
            np.percentile(col, 75),
        ])
    
    return np.array(stats)


def load_data(keypoints_dir: str):
    """Dataset yükle ve özellik çıkar"""
    
    metadata_path = os.path.join(keypoints_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    X = []
    y = []
    
    for item in metadata:
        if item['status'] != 'success':
            continue
        
        video_path = item['video_path']
        keypoint_path = item['keypoint_path']
        
        # Sadece front videoları
        if '/front/' not in video_path.lower():
            continue
        
        # Label
        if 'normal' in video_path.lower():
            label = 0
        elif 'scoliosis' in video_path.lower():
            label = 1
        else:
            continue
        
        # Keypoint yükle
        if not os.path.exists(keypoint_path):
            continue
            
        keypoints = np.load(keypoint_path)
        
        # Özellik çıkar
        try:
            features = extract_posture_features(keypoints)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Hata: {keypoint_path} - {e}")
            continue
    
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("🔬 Klasik ML ile Skolyoz Analizi")
    print("=" * 60)
    
    # Data yükle
    print("\n📊 Veri yükleniyor...")
    X, y = load_data("keypoints")
    
    print(f"  Toplam örnek: {len(X)}")
    print(f"  Normal: {sum(y == 0)}, Scoliosis: {sum(y == 1)}")
    print(f"  Özellik sayısı: {X.shape[1]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Modeller
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', 
                       class_weight='balanced', probability=True))
        ]),
        "SVM (Linear)": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0, 
                       class_weight='balanced', probability=True))
        ]),
    }
    
    print("\n" + "=" * 60)
    print("🎯 Model Eğitimi ve Değerlendirme")
    print("=" * 60)
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        # Train ve test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"  Test Accuracy: {acc:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    Normal doğru: {cm[0,0]}, Normal yanlış: {cm[0,1]}")
        print(f"    Scoliosis yanlış: {cm[1,0]}, Scoliosis doğru: {cm[1,1]}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
    
    # En iyi model detayları
    print("\n" + "=" * 60)
    print(f"🏆 En İyi Model: {best_name}")
    print(f"   Accuracy: {best_acc:.2%}")
    print("=" * 60)
    
    # Detaylı rapor
    y_pred_best = best_model.predict(X_test)
    print("\n📋 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Scoliosis']))
    
    # Model kaydet
    import joblib
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, "saved_models/classical_model.pkl")
    print(f"\n✅ Model kaydedildi: saved_models/classical_model.pkl")
    
    # Feature importance (Random Forest için)
    if "Random Forest" in best_name:
        rf = best_model
        feature_names = []
        base_features = ['shoulder_tilt', 'hip_tilt', 'spine_deviation', 'head_deviation',
                        'shoulder_width', 'hip_width', 'torso_asymmetry', 'diagonal_diff',
                        'ear_tilt', 'abs_shoulder_tilt', 'abs_hip_tilt', 'abs_spine_deviation']
        stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for feat in base_features:
            for stat in stats:
                feature_names.append(f"{feat}_{stat}")
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n📊 En Önemli 10 Özellik:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()

