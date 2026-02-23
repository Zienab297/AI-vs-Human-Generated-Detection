# AI vs. Human-Generated Image Classification

A deep learning pipeline for binary image classification that distinguishes between AI-generated and human-created images, built with PyTorch and EfficientNet-B4.

---

## Overview

This project fine-tunes a pretrained **EfficientNet-B4** model on the [AI vs. Human Generated Dataset](https://www.kaggle.com/datasets/ai-vs-human-generated-dataset) from Kaggle. It covers the full ML workflow — data loading, augmentation, model training, validation, and inference — and generates a submission-ready CSV for the Kaggle competition.

**Task:** Binary classification (`0` = Human-generated, `1` = AI-generated)  
**Platform:** Kaggle (GPU-accelerated, CUDA)

---

## Results

| Metric | Score |
|---|---|
| Final Train Accuracy | **98.74%** |
| Final Validation Accuracy | **98.36%** |
| Final Validation F1 Score | **0.9837** |
| Train Loss (Epoch 10) | 0.0341 |
| Validation Loss (Epoch 10) | 0.0441 |

Training ran for **10 epochs** with a batch size of 32.

---

## Dataset

| Split | Samples |
|---|---|
| Training | 79,950 |
| Test | 5,540 |

**Directory structure expected:**
```
/kaggle/input/ai-vs-human-generated-dataset/
├── train.csv
├── test.csv
├── train_data/
└── test_data_v2/
```

`train.csv` contains columns `file_name` and `label`. `test.csv` contains an `id` column with image paths.

---

## Model Architecture

- **Backbone:** EfficientNet-B4 (pretrained on ImageNet)
- **Classifier head:** Modified final linear layer → 2 output classes (binary)
- **Weights loaded from:** `torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1`

---

## Pipeline

### 1. Data Augmentation

**Training transforms:**
- Resize to 255×255
- Random horizontal flip
- Random rotation (±20°)
- Random resized crop to 224×224 (scale 0.8–1.2)
- Normalize with ImageNet mean/std

**Validation & Test transforms:**
- Resize to 224×224
- Normalize with ImageNet mean/std

### 2. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 5e-5 |
| LR Scheduler | StepLR (γ = 0.7) |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 32 |
| Random Seed | 42 |
| Train/Val Split | 80% / 20% (stratified) |

### 3. Evaluation

Validation is performed after each epoch and reports loss, accuracy, and **macro F1 score**.

### 4. Inference & Submission

The trained model generates predictions on the test set and exports a `submission.csv` file formatted for Kaggle submission.

---

## Dependencies

Install all required packages with:

```bash
pip install timm torch torchvision albumentations opencv-python pandas scikit-learn matplotlib seaborn tqdm pillow
```

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training and image transforms |
| `timm` | Additional pretrained model support |
| `albumentations` | Advanced image augmentation |
| `scikit-learn` | Train/val split, F1 scoring |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Loss/accuracy visualization |
| `tqdm` | Training progress bars |
| `Pillow` / `opencv-python` | Image loading |


