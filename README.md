# Scoliosis Analysis Project

Advanced deep learning-based scoliosis analysis system. Extracts pose keypoints from RGB video files using MediaPipe and performs scoliosis/normal classification using LSTM/Transformer models.

## 📁 Project Structure

```
scoliosis_project/
│
├── dataset/
│   ├── scoliosis/
│   │   ├── front/
│   │   └── side/
│   ├── normal/
│   │   ├── front/
│   │   └── side/
│   └── raw_videos/          # Original videos (optional)
│
├── keypoints/               # Extracted pose keypoint NPY files
│
├── src/
│   ├── extract_keypoints.py # MediaPipe keypoint extraction
│   ├── dataset.py           # Keypoint dataset loader
│   ├── model.py             # LSTM / Transformer models
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation + metrics
│   ├── predict.py           # Single video prediction
│   └── utils.py             # Utility functions
│
├── saved_models/
│   └── best_model.pth
│
├── notebooks/
│   └── EDA.ipynb            # Exploratory analysis, visualizations
│
├── requirements.txt
│
└── README.md
```

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset Structure

Organize your videos in the following structure:

```
dataset/
├── normal/
│   ├── front/    # Normal - front view videos
│   └── side/     # Normal - side view videos
└── scoliosis/
    ├── front/    # Scoliosis - front view videos
    └── side/     # Scoliosis - side view videos
```

**Supported video formats:** AVI, MP4, MOV, MKV, WMV, FLV

## 📊 Step-by-Step Usage Guide

### Step 1: Extract Keypoints from Videos

Extract pose keypoints from all videos in your dataset:

```bash
cd /kullanici_yedek/helin.saygili/Scoliosis_Posture_Disorder_Analysis
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints
```

**What this does:**
- Scans all videos in `dataset/normal/` and `dataset/scoliosis/` folders
- Extracts 33 pose keypoints per frame using MediaPipe
- Saves keypoints as `.npy` files in `keypoints/` directory
- Creates `metadata.json` with video information

**Incremental Mode (Default - Time Saving):**

By default, the script runs in **incremental mode**, which means:
- ✅ Only processes **new videos** that don't have keypoint files yet
- ✅ Skips videos that are already processed
- ✅ Updates metadata automatically
- ⏱️ **Saves significant time** when adding new videos to dataset

**First run (all videos):**
```bash
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints
```

**Adding new videos (incremental - default):**
```bash
# After adding new videos to dataset, run again - only new ones will be processed
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints
# or explicitly:
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints --incremental
```

**Force re-processing all videos:**
```bash
# Process all videos again (even if keypoints exist)
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints --force
```

**Expected output (incremental mode):**
```
🔍 Video dosyaları aranıyor...
📹 Toplam 225 video bulundu
  Normal - Front: 57
  Normal - Side: 56
  Scoliosis - Front: 56
  Scoliosis - Side: 56
📂 Mevcut 219 keypoint dosyası bulundu
🔄 Incremental mode: Sadece yeni videolar işlenecek
⏭️  219 video atlandı (zaten işlenmiş)
🆕 6 yeni video işlenecek
Keypoint çıkarımı: 100%|████████| 6/6 [02:15<00:00, 22.5s/video]

✅ 6 yeni video işlendi
📊 Toplam 225 başarılı keypoint dosyası
📁 Keypoint'ler kaydedildi: keypoints
```

**Expected output (first run):**
```
🔍 Video dosyaları aranıyor...
📹 Toplam 219 video bulundu
  Normal - Front: 55
  Normal - Side: 54
  Scoliosis - Front: 55
  Scoliosis - Side: 55
🔄 Incremental mode: Sadece yeni videolar işlenecek
🆕 219 yeni video işlenecek
Keypoint çıkarımı: 100%|████████| 219/219 [45:30<00:00, 12.45s/video]

✅ 219 yeni video işlendi
📊 Toplam 219 başarılı keypoint dosyası
📁 Keypoint'ler kaydedildi: keypoints
```

**Note:** 
- This step may take 1-3 hours for first run depending on the number and length of videos
- Incremental mode significantly reduces time when adding new videos (only processes new ones)
- Runs on CPU (MediaPipe doesn't use GPU)

---

### Step 2: Train the Model

Train the model using extracted keypoints:

```bash
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --device cuda \
    --save_dir saved_models \
    --model_name best_model
```

**Model Types:**
- `advanced_lstm`: Advanced LSTM with bidirectional layers and attention mechanism (Recommended)
- `transformer`: Transformer encoder model
- `hybrid`: Hybrid LSTM + Transformer model

**GPU Training (Recommended):**
```bash
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --device cuda
```

**CPU Training (if no GPU):**
```bash
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 8 \
    --device cpu
```

**Expected output:**
```
🔧 Device: cuda
📊 Dataset yüklendi:
  Train: 175 örnek
  Test: 44 örnek
  Normal: 110 örnek
  Scoliosis: 109 örnek
📊 Model parametreleri: 1,234,567

🚀 Eğitim başlıyor...
  Model: AdvancedLSTM
  Epochs: 100
  Learning rate: 0.001
  Device: cuda

Epoch 1/100
Training: 100%|████████| 22/22 [00:30<00:00, loss=0.6234, acc=65.23%]
Validation: 100%|████████| 6/6 [00:05<00:00]
Train Loss: 0.6234, Train Acc: 65.23%
Val Loss: 0.5891, Val Acc: 68.18%
✅ Best model kaydedildi! (Val Acc: 68.18%)

...

✅ Eğitim tamamlandı!
  Best validation accuracy: 87.50%
  Model kaydedildi: saved_models/best_model.pth
```

**Training parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 8, use 16-32 for GPU)
- `--device`: Device to use (`cuda` or `cpu`)

---

### Step 3: Evaluate the Model

Evaluate the trained model on the test set:

```bash
python3 src/evaluate.py \
    --keypoints_dir keypoints \
    --model_path saved_models/best_model.pth \
    --model_type advanced_lstm \
    --output_dir results \
    --device cuda
```

**Expected output:**
```
🔧 Device: cuda
📂 Checkpoint yüklendi: saved_models/best_model.pth
  Epoch: 95, Val Acc: 87.50%

🔮 Model değerlendirmesi başlıyor...
Evaluating: 100%|████████| 6/6 [00:10<00:00]

==================================================
📊 DEĞERLENDİRME METRİKLERİ
==================================================

🎯 Genel Metrikler:
  Accuracy:  0.8750
  Precision: 0.8765
  Recall:    0.8750
  F1-Score:  0.8752

📈 Sınıf Bazlı Metrikler:
  Normal:
    Precision: 0.8800
    Recall:    0.8800
    F1-Score:  0.8800
  Scoliosis:
    Precision: 0.8700
    Recall:    0.8700
    F1-Score:  0.8700

🔢 Confusion Matrix:
              Predicted
              Normal  Scoliosis
  Actual Normal      22        3
         Scoliosis    3       16
==================================================

✅ Sonuçlar kaydedildi:
  Metrikler: results/evaluation_metrics.json
  Detaylı sonuçlar: results/detailed_results.json
```

---

### Step 4: Predict on New Videos

Use the trained model to predict scoliosis/normal on new test videos:

#### Single Video Prediction

```bash
python3 src/predict.py \
    --model_path saved_models/best_model.pth \
    --model_type advanced_lstm \
    --video test_video.mp4 \
    --device cuda
```

**Expected output:**
```
🔧 Device: cuda
  GPU: NVIDIA GeForce RTX 3090
📂 Model yükleniyor: saved_models/best_model.pth
📂 Checkpoint yüklendi: saved_models/best_model.pth
  Epoch: 95, Val Acc: 87.50%

🎬 Video işleniyor: test_video.mp4
İşlenen frame sayısı: 30
İşlenen frame sayısı: 60
...

==================================================
📊 TAHMİN SONUCU
==================================================
Video: test_video.mp4
Tahmin: Skolyoz
Güven Skorları:
  Normal:   0.1234 (12.34%)
  Skolyoz:  0.8766 (87.66%)
Frame sayısı: 150
==================================================
```

#### Batch Prediction (Multiple Videos)

```bash
python3 src/predict.py \
    --model_path saved_models/best_model.pth \
    --model_type advanced_lstm \
    --video_dir test_videos/ \
    --output results/predictions.json \
    --device cuda
```

**Save prediction results to JSON:**
```bash
python3 src/predict.py \
    --model_path saved_models/best_model.pth \
    --model_type advanced_lstm \
    --video test_video.mp4 \
    --output results/prediction_result.json \
    --device cuda
```

**JSON output format:**
```json
{
  "video_path": "test_video.mp4",
  "prediction": "Scoliosis",
  "prediction_class": 1,
  "confidence": {
    "Normal": 0.1234,
    "Scoliosis": 0.8766
  },
  "raw_probabilities": [0.1234, 0.8766],
  "num_frames": 150
}
```

---

## 🎯 Features

- ✅ **MediaPipe Pose Detection**: 33 keypoint extraction per frame
- ✅ **Incremental Processing**: Only processes new videos (saves time when adding data)
- ✅ **Advanced Models**: LSTM, Transformer, Hybrid architectures
- ✅ **Multi-view Support**: Front and side view support
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **GPU Support**: CUDA acceleration
- ✅ **Batch Processing**: Process multiple videos efficiently

## 📈 Metrics and Results

Evaluation results are saved in `results/` directory:
- `evaluation_metrics.json`: Overall metrics (accuracy, precision, recall, F1)
- `detailed_results.json`: Detailed prediction results for each sample

## 🔧 Advanced Usage

### Custom Model Parameters

You can customize model parameters in `src/model.py`:

```python
model = build_model(
    model_type="advanced_lstm",
    input_dim=99,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3,
    bidirectional=True,
    use_attention=True
)
```

### Adjust Training Parameters

```bash
# Larger batch size for GPU
python3 src/train.py --batch_size 32 --lr 0.0005

# More epochs
python3 src/train.py --epochs 200

# Different learning rate
python3 src/train.py --lr 0.0001
```

### Check GPU Availability

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 📝 Notes

- **Keypoint extraction** may take 1-3 hours depending on video count and length (runs on CPU)
- **GPU usage is highly recommended** for training (significantly reduces training time)
- Model checkpoints are saved in `saved_models/` directory
- Best model is automatically saved based on validation accuracy
- Training history is saved as JSON for visualization

## 🚀 Quick Start Commands

**Complete workflow:**

```bash
# 1. Extract keypoints (first time - all videos)
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints

# 1b. Add new videos (incremental - only new ones processed)
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints

# 2. Train model
python3 src/train.py --keypoints_dir keypoints --model_type advanced_lstm --epochs 100 --device cuda

# 3. Evaluate model
python3 src/evaluate.py --keypoints_dir keypoints --model_path saved_models/best_model.pth --model_type advanced_lstm

# 4. Predict on new video
python3 src/predict.py --model_path saved_models/best_model.pth --model_type advanced_lstm --video test_video.mp4 --device cuda
```

## 🤝 Contributing

This project is under active development. Feel free to open issues for suggestions or improvements.

## 📄 License

This project is for research purposes.