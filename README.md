# Scoliosis Posture Disorder Analysis

Advanced deep learning-based scoliosis analysis system that extracts pose keypoints from RGB video files using MediaPipe and performs scoliosis/normal classification using LSTM models with attention mechanisms.

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [Advanced Usage](#-advanced-usage)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- ✅ **MediaPipe Pose Detection**: 33 keypoint extraction per frame
- ✅ **Incremental Processing**: Only processes new videos (saves time when adding data)
- ✅ **Advanced LSTM Models**: Bidirectional LSTM with attention mechanism
- ✅ **Multi-view Support**: Front and side view video support
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ **Early Stopping**: Prevents overfitting during training
- ✅ **GPU Support**: CUDA acceleration for faster training
- ✅ **Person-based Split**: Prevents data leakage in train/test split
- ✅ **Data Augmentation**: Gaussian noise, translation, scaling, temporal subsampling

## 📁 Project Structure

```
Scoliosis_Posture_Disorder_Analysis/
│
├── dataset/                    # Video dataset
│   ├── normal/
│   │   ├── front/             # Normal - front view videos
│   │   └── side/              # Normal - side view videos
│   └── scoliosis/
│       ├── front/             # Scoliosis - front view videos
│       └── side/              # Scoliosis - side view videos
│
├── keypoints/                  # Extracted pose keypoint NPY files
│   └── metadata.json          # Video metadata
│
├── src/
│   ├── extract_keypoints.py   # MediaPipe keypoint extraction
│   ├── dataset.py             # Keypoint dataset loader
│   ├── model.py               # LSTM models with attention
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation + metrics
│   ├── predict.py             # Single video prediction
│   ├── utils.py               # Utility functions
│   ├── find_best_seed.py      # Seed optimization
│   └── train_classical.py     # Classical ML models (optional)
│
├── saved_models/
│   ├── new_model_fixed_v2.pth # Best front-view model (86% accuracy)
│   ├── side_version.pth       # Best multi-view model (90% accuracy)
│   └── *_history.json         # Training history files
│
├── results/                    # Evaluation results
│   └── evaluation_metrics.json
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Scoliosis_Posture_Disorder_Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For CUDA support, ensure you have the appropriate PyTorch version installed. If you need CUDA 11.7, the requirements.txt includes `torch==1.13.1+cu117`. For other CUDA versions, visit [PyTorch Installation](https://pytorch.org/get-started/locally/).

### 3. Prepare Dataset Structure

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

**Supported video formats**: AVI, MP4, MOV, MKV, WMV, FLV

## 🎯 Quick Start

Complete workflow in 4 steps:

```bash
# 1. Extract keypoints from videos
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints

# 2. Train model (front view only)
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --device cuda \
    --save_dir saved_models \
    --model_name my_model

# 3. Train model (front + side views)
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --device cuda \
    --include_side \
    --save_dir saved_models \
    --model_name side_version

# 4. Evaluate model
python3 src/evaluate.py \
    --keypoints_dir keypoints \
    --model_path saved_models/my_model.pth \
    --model_type advanced_lstm \
    --output_dir results \
    --device cuda

# 5. Predict on new video
python3 src/predict.py \
    --model_path saved_models/my_model.pth \
    --model_type advanced_lstm \
    --video test_video.mp4 \
    --device cuda
```

## 📖 Usage Guide

### Step 1: Extract Keypoints from Videos

Extract pose keypoints from all videos in your dataset:

```bash
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

**Force re-processing all videos:**
```bash
python3 src/extract_keypoints.py --dataset_root dataset --output_dir keypoints --force
```

**Note:** 
- This step may take 1-3 hours for first run depending on the number and length of videos
- Incremental mode significantly reduces time when adding new videos
- Runs on CPU (MediaPipe doesn't use GPU)

---

### Step 2: Train the Model

Train the model using extracted keypoints:

#### Front View Only (Recommended for starting)

```bash
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --device cuda \
    --save_dir saved_models \
    --model_name my_model
```

#### Front + Side Views (Best Performance)

```bash
python3 src/train.py \
    --keypoints_dir keypoints \
    --model_type advanced_lstm \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 16 \
    --device cuda \
    --include_side \
    --save_dir saved_models \
    --model_name side_version
```

**Training Parameters:**
- `--keypoints_dir`: Directory containing extracted keypoints (default: `keypoints`)
- `--model_type`: Model architecture - `advanced_lstm` (recommended), `simple_lstm`, `transformer`, `hybrid`
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 16 for GPU, use 8 for CPU)
- `--device`: Device to use (`cuda` or `cpu`)
- `--include_side`: Include side view videos in training
- `--save_dir`: Directory to save models (default: `saved_models`)
- `--model_name`: Name for the saved model (default: `best_model`)
- `--seed`: Random seed for reproducibility (default: 42)

**Expected output:**
```
🎲 Random seed sabitlendi: 42
👥 Person-based split kullanılıyor (overfitting önleme)...
📊 Dataset yüklendi:
  Train: 196 örnek (Normal: 104, Scoliosis: 92)
  Test: 50 örnek (Normal: 27, Scoliosis: 23)
🚀 Eğitim başlıyor...
  Model: AdvancedLSTM
  Epochs: 100
  Learning rate: 0.001
  Device: cuda

Epoch 1/100
Training: 100%|████████| 13/13 [00:15<00:00, loss=0.6234, acc=65.23%]
Validation: 100%|████████| 4/4 [00:05<00:00]
Train Loss: 0.6234, Train Acc: 65.23%
Val Loss: 0.5891, Val Acc: 68.00%
✅ Best model kaydedildi! (Val Acc: 68.00%)

...

✅ Eğitim tamamlandı!
  Best validation accuracy: 90.00%
  Model kaydedildi: saved_models/side_version.pth
```

---

### Step 3: Evaluate the Model

Evaluate the trained model on the test set:

```bash
python3 src/evaluate.py \
    --keypoints_dir keypoints \
    --model_path saved_models/side_version.pth \
    --model_type advanced_lstm \
    --output_dir results \
    --device cuda
```

**For side view models, include the flag:**
```bash
python3 src/evaluate.py \
    --keypoints_dir keypoints \
    --model_path saved_models/side_version.pth \
    --model_type advanced_lstm \
    --output_dir results \
    --device cuda \
    --include_side
```

**Expected output:**
```
📂 Checkpoint yüklendi: saved_models/side_version.pth
  Epoch: 29, Val Acc: 90.00%

Evaluating: 100%|████████| 4/4 [00:10<00:00]

==================================================
📊 EVALUATION METRICS
==================================================

🎯 Overall Metrics:
  Accuracy:  0.9000 (90.00%)
  Precision: 0.9003 (90.03%)
  Recall:    0.9000 (90.00%)
  F1-Score:  0.8998 (89.98%)

📈 Per-Class Metrics:
  Normal:
    Precision: 0.8929 (89.29%)
    Recall:    0.9259 (92.59%)
    F1-Score:  0.9091 (90.91%)
  Scoliosis:
    Precision: 0.9091 (90.91%)
    Recall:    0.8696 (86.96%)
    F1-Score:  0.8889 (88.89%)

🔢 Confusion Matrix:
              Predicted
              Normal  Scoliosis
  Actual Normal      25        2
         Scoliosis    3       20
==================================================

✅ Results saved:
  Metrics: results/evaluation_metrics.json
  Detailed results: results/detailed_results.json
```

---

### Step 4: Predict on New Videos

Use the trained model to predict scoliosis/normal on new test videos:

#### Single Video Prediction

```bash
python3 src/predict.py \
    --model_path saved_models/side_version.pth \
    --model_type advanced_lstm \
    --video test_video.mp4 \
    --device cuda
```

#### Batch Prediction (Multiple Videos)

```bash
python3 src/predict.py \
    --model_path saved_models/side_version.pth \
    --model_type advanced_lstm \
    --video_dir test_videos/ \
    --output results/predictions.json \
    --device cuda
```

**Expected output:**
```
📂 Model loaded: saved_models/side_version.pth
🎬 Processing video: test_video.mp4
Processing frames: 30, 60, 90...

==================================================
📊 PREDICTION RESULT
==================================================
Video: test_video.mp4
Prediction: Scoliosis
Confidence Scores:
  Normal:   0.1234 (12.34%)
  Scoliosis: 0.8766 (87.66%)
Number of frames: 150
==================================================
```

**Save prediction results to JSON:**
```bash
python3 src/predict.py \
    --model_path saved_models/side_version.pth \
    --model_type advanced_lstm \
    --video test_video.mp4 \
    --output results/prediction_result.json \
    --device cuda
```

---

## 📊 Model Performance

### Available Models

1. **`new_model_fixed_v2.pth`** (Front View Only)
   - **Test Accuracy**: 86.11%
   - **Validation Accuracy**: 86.11%
   - **Best Epoch**: 36
   - **Dataset**: Front view videos only
   - **Use Case**: When only front view videos are available

2. **`side_version.pth`** (Front + Side Views) ⭐ **Best Performance**
   - **Test Accuracy**: 90.00%
   - **Validation Accuracy**: 90.00%
   - **Best Epoch**: 29
   - **Dataset**: Front + Side view videos
   - **Configuration**:
     - Hidden Dimension: 192
     - Number of Layers: 2
     - Dropout: 0.25
     - Attention: Enabled
     - Seed: 42
   - **Use Case**: Recommended when both front and side view videos are available

### Performance Comparison

| Model | Test Accuracy | Precision | Recall | F1-Score | Views |
|-------|--------------|-----------|--------|----------|-------|
| `new_model_fixed_v2.pth` | 86.11% | 86.20% | 86.11% | 86.15% | Front only |
| `side_version.pth` | **90.00%** | **90.03%** | **90.00%** | **89.98%** | Front + Side |

---

## 🔧 Advanced Usage

### Find Best Random Seed

Optimize the random seed for better performance:

```bash
python3 src/find_best_seed.py \
    --seeds "42,123,456,789,1111,2222,3333,4444" \
    --model_type advanced_lstm \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 16 \
    --output results/best_seed_results.json
```

### Custom Model Parameters

You can customize model parameters in `src/model.py`:

```python
model = build_model(
    model_type="advanced_lstm",
    input_dim=99,
    hidden_dim=192,
    num_layers=2,
    dropout=0.25,
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

# Custom seed
python3 src/train.py --seed 123
```

### Check GPU Availability

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8` or `--batch_size 4`
   - Use CPU: `--device cpu`

2. **MediaPipe Installation Issues**
   - Ensure Python version is 3.8+
   - Try: `pip install --upgrade mediapipe`

3. **Keypoint Extraction is Slow**
   - This is normal - MediaPipe runs on CPU
   - First run may take 1-3 hours depending on dataset size
   - Use incremental mode for subsequent runs

4. **Model Not Improving**
   - Try different random seeds: `--seed 123`
   - Adjust learning rate: `--lr 0.0005` or `--lr 0.0015`
   - Include side views: `--include_side`
   - Check if dataset is balanced

5. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python3 --version` (should be 3.8+)

---

## 📝 Notes

- **Keypoint extraction** may take 1-3 hours depending on video count and length (runs on CPU)
- **GPU usage is highly recommended** for training (significantly reduces training time)
- Model checkpoints are saved in `saved_models/` directory
- Best model is automatically saved based on validation accuracy
- Training history is saved as JSON for visualization
- Person-based split ensures no data leakage between train and test sets
- Early stopping prevents overfitting (patience: 30 epochs)

---

## 📄 License

This project is for research purposes.

---

## 🤝 Contributing

This project is under active development. Feel free to open issues for suggestions or improvements.
