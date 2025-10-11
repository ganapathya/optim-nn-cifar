# CIFAR-100 ResNet-18 Training

ResNet-18 implementation for CIFAR-100 dataset, targeting 73%+ top-1 accuracy.

## 🎯 Project Goal

Train a ResNet-18 model from scratch on CIFAR-100 to achieve **73% top-1 accuracy**.

## 📊 Quick Start

### 1. Test Model Architecture

```bash
cd cifar100
python model.py
```

This will display the ResNet-18 architecture and parameter count (~11M parameters).

### 2. Test Data Loaders

```bash
python utils.py
```

This will show the augmentation details and test the data loading pipeline.

### 3. Start Training

```bash
python train.py
```

Training will run for 100 epochs (approximately 2-3 hours on Apple Silicon).

### 4. Test Inference

After training, test the model:

```bash
python inference.py logs/cifar100/best_model.pth
```

## 📁 Files Overview

- **`config.py`**: All hyperparameters and configuration in one place
- **`model.py`**: ResNet-18 architecture adapted for CIFAR-100
- **`utils.py`**: Data loaders with albumentations augmentations
- **`train.py`**: Complete training pipeline with logging
- **`inference.py`**: Inference utilities for trained model
- **`README.md`**: This file

## 🏗️ Model Architecture

**ResNet-18 for CIFAR-100:**

- Input: 3×32×32 RGB images
- Modified first conv: 3×3 (stride=1) instead of 7×7 (stride=2)
- No initial maxpool (32×32 images are too small)
- 4 residual blocks: [2, 2, 2, 2] layers
- Feature sizes: 64 → 128 → 256 → 512
- Global average pooling
- Fully connected: 512 → 100 classes
- **Total parameters:** ~11 million

## 🔧 Training Configuration

### Optimizer

- **Type:** SGD with Nesterov momentum
- **Initial LR:** 0.1
- **Momentum:** 0.9
- **Weight Decay:** 5e-4

### Learning Rate Schedule

- **Warmup:** 5 epochs (linear warmup)
- **Main Schedule:** Cosine annealing to 0
- **Total Epochs:** 100

### Loss Function

- **CrossEntropyLoss** with label smoothing (0.1)
- Helps prevent overconfident predictions
- Improves generalization

### Data Augmentation

1. **Random Crop:** 32×32 with padding=4
2. **Random Horizontal Flip:** p=0.5
3. **Cutout:** 16×16 patch removal, p=0.5
4. **Normalization:** CIFAR-100 mean/std

### Regularization

- Weight decay: 5e-4
- Label smoothing: 0.1
- Batch normalization in ResNet blocks
- Dropout: None (ResNet doesn't typically use it)

## 📈 Expected Results

**Target:** 73% top-1 accuracy

**Typical Learning Curve:**

- Epoch 10: ~35-40% accuracy
- Epoch 30: ~55-60% accuracy
- Epoch 50: ~65-68% accuracy
- Epoch 80: ~70-72% accuracy
- Epoch 100: ~73-75% accuracy

## 📊 CIFAR-100 Dataset

- **Training samples:** 50,000
- **Test samples:** 10,000
- **Classes:** 100 (fine-grained)
- **Superclasses:** 20
- **Image size:** 32×32 RGB
- **Normalization:**
  - Mean: (0.5071, 0.4867, 0.4408)
  - Std: (0.2675, 0.2565, 0.2761)

### Class Categories

Animals, vehicles, household items, food, plants, natural scenes, insects, aquatic creatures, and more. See `utils.py` for complete list.

## 🔄 Training Process

### What Happens During Training

1. **Data Loading:** CIFAR-100 is automatically downloaded to `data/` folder
2. **Model Initialization:** ResNet-18 with random weights (Kaiming initialization)
3. **Training Loop:** 100 epochs with validation after each epoch
4. **Checkpointing:** Best model saved, checkpoints every 10 epochs
5. **Logging:** Detailed logs saved to `logs/cifar100/`

### Output Files

After training:

```
logs/cifar100/
├── training_log_YYYYMMDD_HHMMSS.txt  # Detailed training log
├── best_model.pth                     # Best model checkpoint
├── checkpoint_epoch_10.pth            # Periodic checkpoints
├── checkpoint_epoch_20.pth
└── ...
```

## 💾 Using Trained Model

### Load for Inference

```python
from cifar100.inference import load_predictor
from PIL import Image

# Load predictor
predictor = load_predictor('logs/cifar100/best_model.pth')

# Predict single image
image = Image.open('test_image.jpg')
results = predictor.predict(image, top_k=5)

# Display results
for class_name, class_id, probability in results:
    print(f"{class_name}: {probability:.2%}")
```

### Batch Prediction

```python
images = [Image.open(f'image_{i}.jpg') for i in range(10)]
batch_results = predictor.predict_batch(images, top_k=5)

for i, results in enumerate(batch_results):
    print(f"\nImage {i}:")
    for class_name, class_id, prob in results:
        print(f"  {class_name}: {prob:.2%}")
```

## 🛠️ Customization

### Modify Hyperparameters

Edit `config.py` to change:

- Batch size
- Learning rate
- Number of epochs
- Augmentation parameters
- Weight decay
- Label smoothing

### Try Different Architectures

In `model.py`, you can also try:

- `resnet34()` for a larger model (better accuracy, slower training)

## 🐛 Troubleshooting

### Out of Memory

- Reduce batch size in `config.py` (try 64 or 32)
- Reduce number of workers to 0

### Slow Training

- Ensure MPS (Apple Silicon GPU) is being used
- Check `USE_MPS = True` in `config.py`
- Close other applications

### Model Not Improving

- Check learning rate (might need adjustment)
- Verify augmentations are working (`python utils.py`)
- Increase training epochs

## 📚 References

1. **ResNet Paper:** He et al., "Deep Residual Learning for Image Recognition" (2016)
2. **CIFAR-100:** Krizhevsky & Hinton, "Learning Multiple Layers of Features from Tiny Images" (2009)
3. **Label Smoothing:** Szegedy et al., "Rethinking the Inception Architecture" (2016)

## 🎓 Key Techniques Used

1. **Residual Learning:** Skip connections for deeper networks
2. **Cosine Annealing:** Smooth learning rate decay
3. **Warmup:** Gradual LR increase at start
4. **Label Smoothing:** Soften target distribution
5. **Strong Augmentation:** Random crop, flip, cutout
6. **Weight Decay:** L2 regularization
7. **Batch Normalization:** Stabilize training

## ✅ Success Criteria

- [x] ResNet-18 architecture implemented
- [x] CIFAR-100 data pipeline with augmentations
- [x] Training script with proper regularization
- [x] Validation after each epoch
- [x] Model checkpointing
- [x] Inference utilities
- [ ] Achieve 73%+ test accuracy (after training)

## 🚀 Next Steps After Training

1. Check `logs/cifar100/training_log_*.txt` for training history
2. Verify accuracy meets target (73%+)
3. Test inference with `inference.py`
4. Deploy to Huggingface Spaces using `../gradio_app/`

---

# Training Logs

CIFAR-100 ResNet-18 TRAINING LOG
Start Time: 2025-10-11 10:36:20
Device: mps
Model: ResNet-18
Target Accuracy: 73.0%
================================================================================

Starting training for 100 epochs...
Batch size: 128
Initial learning rate: 0.1
Weight decay: 0.0005
Label smoothing: 0.1
Warmup epochs: 5
================================================================================

Epoch [ 1/100] | Time: 39.0s | LR: 0.020000 | Train Loss: 4.2847 | Train Acc: 5.91% | Test Loss: 3.9861 | Test Acc: 11.14% <- Best!
Epoch [ 2/100] | Time: 38.0s | LR: 0.040000 | Train Loss: 3.7553 | Train Acc: 15.39% | Test Loss: 3.9228 | Test Acc: 17.50% <- Best!
Epoch [ 3/100] | Time: 38.2s | LR: 0.060000 | Train Loss: 3.3864 | Train Acc: 23.46% | Test Loss: 3.3992 | Test Acc: 26.19% <- Best!
Epoch [ 4/100] | Time: 38.0s | LR: 0.080000 | Train Loss: 3.0946 | Train Acc: 30.29% | Test Loss: 3.0460 | Test Acc: 33.33% <- Best!
Epoch [ 5/100] | Time: 38.7s | LR: 0.100000 | Train Loss: 2.8616 | Train Acc: 36.70% | Test Loss: 2.7924 | Test Acc: 38.98% <- Best!
Epoch [ 6/100] | Time: 41.9s | LR: 0.100000 | Train Loss: 2.6753 | Train Acc: 41.60% | Test Loss: 2.6697 | Test Acc: 42.62% <- Best!
Epoch [ 7/100] | Time: 50.9s | LR: 0.099973 | Train Loss: 2.5259 | Train Acc: 45.61% | Test Loss: 2.5579 | Test Acc: 45.58% <- Best!
Epoch [ 8/100] | Time: 76.0s | LR: 0.099891 | Train Loss: 2.4326 | Train Acc: 48.31% | Test Loss: 2.4152 | Test Acc: 49.24% <- Best!
Epoch [ 9/100] | Time: 72.5s | LR: 0.099754 | Train Loss: 2.3666 | Train Acc: 50.15% | Test Loss: 2.3587 | Test Acc: 51.07% <- Best!
Epoch [ 10/100] | Time: 55.3s | LR: 0.099563 | Train Loss: 2.3146 | Train Acc: 51.90% | Test Loss: 2.4071 | Test Acc: 49.37%
Epoch [ 11/100] | Time: 49.3s | LR: 0.099318 | Train Loss: 2.2733 | Train Acc: 53.00% | Test Loss: 2.3902 | Test Acc: 50.65%
Epoch [ 12/100] | Time: 46.5s | LR: 0.099019 | Train Loss: 2.2333 | Train Acc: 54.30% | Test Loss: 2.2381 | Test Acc: 54.51% <- Best!
Epoch [ 13/100] | Time: 73.5s | LR: 0.098666 | Train Loss: 2.2111 | Train Acc: 54.92% | Test Loss: 2.2954 | Test Acc: 53.12%
Epoch [ 14/100] | Time: 42.0s | LR: 0.098260 | Train Loss: 2.1821 | Train Acc: 55.87% | Test Loss: 2.3187 | Test Acc: 51.60%
Epoch [ 15/100] | Time: 43.3s | LR: 0.097802 | Train Loss: 2.1627 | Train Acc: 56.43% | Test Loss: 2.2198 | Test Acc: 55.00% <- Best!
Epoch [ 16/100] | Time: 43.9s | LR: 0.097291 | Train Loss: 2.1309 | Train Acc: 57.35% | Test Loss: 2.1736 | Test Acc: 56.59% <- Best!
Epoch [ 17/100] | Time: 69.6s | LR: 0.096728 | Train Loss: 2.1151 | Train Acc: 57.89% | Test Loss: 2.4025 | Test Acc: 49.82%
Epoch [ 18/100] | Time: 137.7s | LR: 0.096114 | Train Loss: 2.0950 | Train Acc: 58.67% | Test Loss: 2.2758 | Test Acc: 53.05%
Epoch [ 19/100] | Time: 64.0s | LR: 0.095450 | Train Loss: 2.0856 | Train Acc: 58.97% | Test Loss: 2.2608 | Test Acc: 54.43%
Epoch [ 20/100] | Time: 51.9s | LR: 0.094736 | Train Loss: 2.0677 | Train Acc: 59.28% | Test Loss: 2.4285 | Test Acc: 49.45%
Epoch [ 21/100] | Time: 49.8s | LR: 0.093974 | Train Loss: 2.0571 | Train Acc: 59.73% | Test Loss: 2.1017 | Test Acc: 58.18% <- Best!
Epoch [ 22/100] | Time: 48.4s | LR: 0.093163 | Train Loss: 2.0338 | Train Acc: 60.31% | Test Loss: 2.1852 | Test Acc: 57.40%
Epoch [ 23/100] | Time: 46.4s | LR: 0.092305 | Train Loss: 2.0241 | Train Acc: 60.50% | Test Loss: 2.1265 | Test Acc: 58.00%
Epoch [ 24/100] | Time: 45.4s | LR: 0.091400 | Train Loss: 2.0054 | Train Acc: 61.48% | Test Loss: 2.1748 | Test Acc: 57.14%
Epoch [ 25/100] | Time: 45.3s | LR: 0.090451 | Train Loss: 1.9951 | Train Acc: 61.63% | Test Loss: 2.0633 | Test Acc: 60.09% <- Best!
Epoch [ 26/100] | Time: 50.6s | LR: 0.089457 | Train Loss: 1.9797 | Train Acc: 62.08% | Test Loss: 2.1213 | Test Acc: 58.40%
Epoch [ 27/100] | Time: 55.3s | LR: 0.088420 | Train Loss: 1.9777 | Train Acc: 61.95% | Test Loss: 2.2385 | Test Acc: 55.08%
Epoch [ 28/100] | Time: 53.9s | LR: 0.087341 | Train Loss: 1.9604 | Train Acc: 62.77% | Test Loss: 2.1541 | Test Acc: 58.10%
Epoch [ 29/100] | Time: 51.8s | LR: 0.086221 | Train Loss: 1.9507 | Train Acc: 62.81% | Test Loss: 2.1846 | Test Acc: 56.18%
Epoch [ 30/100] | Time: 50.0s | LR: 0.085062 | Train Loss: 1.9375 | Train Acc: 63.16% | Test Loss: 2.2026 | Test Acc: 57.47%
Epoch [ 31/100] | Time: 48.6s | LR: 0.083864 | Train Loss: 1.9217 | Train Acc: 63.92% | Test Loss: 2.0450 | Test Acc: 60.89% <- Best!
Epoch [ 32/100] | Time: 49.2s | LR: 0.082629 | Train Loss: 1.9141 | Train Acc: 63.98% | Test Loss: 2.0016 | Test Acc: 62.49% <- Best!
Epoch [ 33/100] | Time: 49.4s | LR: 0.081359 | Train Loss: 1.9013 | Train Acc: 64.39% | Test Loss: 2.1931 | Test Acc: 56.27%
Epoch [ 34/100] | Time: 48.6s | LR: 0.080054 | Train Loss: 1.8900 | Train Acc: 64.92% | Test Loss: 2.0305 | Test Acc: 61.02%
Epoch [ 35/100] | Time: 47.0s | LR: 0.078716 | Train Loss: 1.8797 | Train Acc: 65.48% | Test Loss: 2.1640 | Test Acc: 57.59%
Epoch [ 36/100] | Time: 47.0s | LR: 0.077347 | Train Loss: 1.8667 | Train Acc: 65.80% | Test Loss: 2.0600 | Test Acc: 61.12%
Epoch [ 37/100] | Time: 47.1s | LR: 0.075948 | Train Loss: 1.8574 | Train Acc: 65.81% | Test Loss: 2.0429 | Test Acc: 60.93%
Epoch [ 38/100] | Time: 47.8s | LR: 0.074521 | Train Loss: 1.8523 | Train Acc: 66.00% | Test Loss: 2.0604 | Test Acc: 60.92%
Epoch [ 39/100] | Time: 50.9s | LR: 0.073067 | Train Loss: 1.8364 | Train Acc: 66.54% | Test Loss: 1.9893 | Test Acc: 62.77% <- Best!
Epoch [ 40/100] | Time: 50.3s | LR: 0.071588 | Train Loss: 1.8177 | Train Acc: 67.20% | Test Loss: 2.0286 | Test Acc: 60.99%
Epoch [ 41/100] | Time: 52.2s | LR: 0.070085 | Train Loss: 1.8053 | Train Acc: 67.46% | Test Loss: 2.0138 | Test Acc: 62.33%
Epoch [ 42/100] | Time: 51.1s | LR: 0.068560 | Train Loss: 1.7980 | Train Acc: 67.70% | Test Loss: 2.0771 | Test Acc: 60.93%
Epoch [ 43/100] | Time: 57.5s | LR: 0.067015 | Train Loss: 1.7873 | Train Acc: 68.22% | Test Loss: 2.1011 | Test Acc: 59.68%
Epoch [ 44/100] | Time: 51.1s | LR: 0.065451 | Train Loss: 1.7723 | Train Acc: 68.57% | Test Loss: 1.9204 | Test Acc: 64.25% <- Best!
Epoch [ 45/100] | Time: 49.4s | LR: 0.063870 | Train Loss: 1.7586 | Train Acc: 68.97% | Test Loss: 1.9152 | Test Acc: 64.74% <- Best!
Epoch [ 46/100] | Time: 49.3s | LR: 0.062274 | Train Loss: 1.7436 | Train Acc: 69.71% | Test Loss: 2.0499 | Test Acc: 60.70%
Epoch [ 47/100] | Time: 50.6s | LR: 0.060665 | Train Loss: 1.7290 | Train Acc: 69.91% | Test Loss: 1.9381 | Test Acc: 64.14%
Epoch [ 48/100] | Time: 48.9s | LR: 0.059044 | Train Loss: 1.7128 | Train Acc: 70.81% | Test Loss: 1.9813 | Test Acc: 62.62%
Epoch [ 49/100] | Time: 48.8s | LR: 0.057413 | Train Loss: 1.7017 | Train Acc: 70.93% | Test Loss: 1.9304 | Test Acc: 64.83% <- Best!
Epoch [ 50/100] | Time: 49.3s | LR: 0.055774 | Train Loss: 1.6898 | Train Acc: 71.26% | Test Loss: 2.0027 | Test Acc: 62.24%
Epoch [ 51/100] | Time: 50.0s | LR: 0.054129 | Train Loss: 1.6731 | Train Acc: 71.81% | Test Loss: 1.9118 | Test Acc: 64.72%
Epoch [ 52/100] | Time: 49.9s | LR: 0.052479 | Train Loss: 1.6492 | Train Acc: 72.63% | Test Loss: 1.9701 | Test Acc: 63.63%
Epoch [ 53/100] | Time: 50.2s | LR: 0.050827 | Train Loss: 1.6500 | Train Acc: 72.60% | Test Loss: 1.9186 | Test Acc: 65.37% <- Best!
Epoch [ 54/100] | Time: 50.2s | LR: 0.049173 | Train Loss: 1.6195 | Train Acc: 73.59% | Test Loss: 1.8610 | Test Acc: 66.67% <- Best!
Epoch [ 55/100] | Time: 52.0s | LR: 0.047521 | Train Loss: 1.5991 | Train Acc: 74.31% | Test Loss: 1.9647 | Test Acc: 63.93%
Epoch [ 56/100] | Time: 50.3s | LR: 0.045871 | Train Loss: 1.5835 | Train Acc: 74.76% | Test Loss: 1.8348 | Test Acc: 68.07% <- Best!
Epoch [ 57/100] | Time: 50.0s | LR: 0.044226 | Train Loss: 1.5668 | Train Acc: 75.23% | Test Loss: 1.8442 | Test Acc: 67.21%
Epoch [ 58/100] | Time: 49.8s | LR: 0.042587 | Train Loss: 1.5563 | Train Acc: 75.81% | Test Loss: 1.9011 | Test Acc: 65.75%
Epoch [ 59/100] | Time: 49.8s | LR: 0.040956 | Train Loss: 1.5324 | Train Acc: 76.53% | Test Loss: 1.8452 | Test Acc: 67.79%
Epoch [ 60/100] | Time: 51.9s | LR: 0.039335 | Train Loss: 1.5142 | Train Acc: 77.14% | Test Loss: 1.8228 | Test Acc: 68.02%
Epoch [ 61/100] | Time: 49.8s | LR: 0.037726 | Train Loss: 1.5004 | Train Acc: 77.70% | Test Loss: 1.8259 | Test Acc: 68.43% <- Best!
Epoch [ 62/100] | Time: 48.3s | LR: 0.036130 | Train Loss: 1.4821 | Train Acc: 78.08% | Test Loss: 1.7556 | Test Acc: 70.22% <- Best!
Epoch [ 63/100] | Time: 49.8s | LR: 0.034549 | Train Loss: 1.4599 | Train Acc: 79.19% | Test Loss: 1.8613 | Test Acc: 67.08%
Epoch [ 64/100] | Time: 55.8s | LR: 0.032985 | Train Loss: 1.4374 | Train Acc: 79.69% | Test Loss: 1.8465 | Test Acc: 66.86%
Epoch [ 65/100] | Time: 54.4s | LR: 0.031440 | Train Loss: 1.4221 | Train Acc: 80.54% | Test Loss: 1.8178 | Test Acc: 68.66%
Epoch [ 66/100] | Time: 53.6s | LR: 0.029915 | Train Loss: 1.3983 | Train Acc: 81.25% | Test Loss: 1.7482 | Test Acc: 70.25% <- Best!
Epoch [ 67/100] | Time: 52.1s | LR: 0.028412 | Train Loss: 1.3763 | Train Acc: 81.94% | Test Loss: 1.7378 | Test Acc: 70.74% <- Best!
Epoch [ 68/100] | Time: 52.6s | LR: 0.026933 | Train Loss: 1.3475 | Train Acc: 82.88% | Test Loss: 1.7436 | Test Acc: 70.56%
Epoch [ 69/100] | Time: 55.2s | LR: 0.025479 | Train Loss: 1.3301 | Train Acc: 83.60% | Test Loss: 1.7105 | Test Acc: 71.58% <- Best!
Epoch [ 70/100] | Time: 50.9s | LR: 0.024052 | Train Loss: 1.3140 | Train Acc: 84.19% | Test Loss: 1.7143 | Test Acc: 71.93% <- Best!
Epoch [ 71/100] | Time: 50.4s | LR: 0.022653 | Train Loss: 1.2922 | Train Acc: 84.80% | Test Loss: 1.6879 | Test Acc: 72.04% <- Best!
Epoch [ 72/100] | Time: 53.2s | LR: 0.021284 | Train Loss: 1.2615 | Train Acc: 86.21% | Test Loss: 1.6782 | Test Acc: 73.13% <- Best!

✓ Target accuracy 73.0% achieved at epoch 72!
Epoch [ 73/100] | Time: 48.9s | LR: 0.019946 | Train Loss: 1.2487 | Train Acc: 86.66% | Test Loss: 1.7290 | Test Acc: 71.42%
Epoch [ 74/100] | Time: 47.9s | LR: 0.018641 | Train Loss: 1.2190 | Train Acc: 87.64% | Test Loss: 1.6797 | Test Acc: 72.76%
Epoch [ 75/100] | Time: 49.8s | LR: 0.017371 | Train Loss: 1.2021 | Train Acc: 88.27% | Test Loss: 1.6391 | Test Acc: 74.22% <- Best!

✓ Target accuracy 73.0% achieved at epoch 75!
Epoch [ 76/100] | Time: 54.5s | LR: 0.016136 | Train Loss: 1.1759 | Train Acc: 89.28% | Test Loss: 1.6591 | Test Acc: 73.56%

✓ Target accuracy 73.0% achieved at epoch 76!
Epoch [ 77/100] | Time: 62.4s | LR: 0.014938 | Train Loss: 1.1545 | Train Acc: 89.93% | Test Loss: 1.6702 | Test Acc: 73.76%

✓ Target accuracy 73.0% achieved at epoch 77!
Epoch [ 78/100] | Time: 65.4s | LR: 0.013779 | Train Loss: 1.1373 | Train Acc: 90.67% | Test Loss: 1.6441 | Test Acc: 74.05%

✓ Target accuracy 73.0% achieved at epoch 78!
Epoch [ 79/100] | Time: 61.6s | LR: 0.012659 | Train Loss: 1.1135 | Train Acc: 91.45% | Test Loss: 1.6260 | Test Acc: 75.01% <- Best!

✓ Target accuracy 73.0% achieved at epoch 79!
Epoch [ 80/100] | Time: 57.4s | LR: 0.011580 | Train Loss: 1.0980 | Train Acc: 91.99% | Test Loss: 1.6113 | Test Acc: 75.56% <- Best!

✓ Target accuracy 73.0% achieved at epoch 80!
Epoch [ 81/100] | Time: 55.5s | LR: 0.010543 | Train Loss: 1.0728 | Train Acc: 92.94% | Test Loss: 1.6068 | Test Acc: 75.17%

✓ Target accuracy 73.0% achieved at epoch 81!
Epoch [ 82/100] | Time: 55.6s | LR: 0.009549 | Train Loss: 1.0558 | Train Acc: 93.40% | Test Loss: 1.5852 | Test Acc: 75.99% <- Best!

✓ Target accuracy 73.0% achieved at epoch 82!
Epoch [ 83/100] | Time: 60.8s | LR: 0.008600 | Train Loss: 1.0405 | Train Acc: 94.06% | Test Loss: 1.5812 | Test Acc: 76.22% <- Best!

✓ Target accuracy 73.0% achieved at epoch 83!
Epoch [ 84/100] | Time: 53.6s | LR: 0.007695 | Train Loss: 1.0238 | Train Acc: 94.53% | Test Loss: 1.5747 | Test Acc: 76.54% <- Best!

✓ Target accuracy 73.0% achieved at epoch 84!
Epoch [ 85/100] | Time: 52.0s | LR: 0.006837 | Train Loss: 1.0057 | Train Acc: 95.15% | Test Loss: 1.5581 | Test Acc: 77.06% <- Best!

✓ Target accuracy 73.0% achieved at epoch 85!
Epoch [ 86/100] | Time: 50.8s | LR: 0.006026 | Train Loss: 0.9928 | Train Acc: 95.43% | Test Loss: 1.5557 | Test Acc: 77.22% <- Best!

✓ Target accuracy 73.0% achieved at epoch 86!
Epoch [ 87/100] | Time: 51.9s | LR: 0.005264 | Train Loss: 0.9763 | Train Acc: 96.27% | Test Loss: 1.5527 | Test Acc: 77.42% <- Best!

✓ Target accuracy 73.0% achieved at epoch 87!
Epoch [ 88/100] | Time: 49.6s | LR: 0.004550 | Train Loss: 0.9678 | Train Acc: 96.36% | Test Loss: 1.5553 | Test Acc: 77.50% <- Best!

✓ Target accuracy 73.0% achieved at epoch 88!
Epoch [ 89/100] | Time: 54.5s | LR: 0.003886 | Train Loss: 0.9584 | Train Acc: 96.58% | Test Loss: 1.5555 | Test Acc: 77.44%

✓ Target accuracy 73.0% achieved at epoch 89!
Epoch [ 90/100] | Time: 53.0s | LR: 0.003272 | Train Loss: 0.9474 | Train Acc: 97.05% | Test Loss: 1.5467 | Test Acc: 77.47%

✓ Target accuracy 73.0% achieved at epoch 90!
Epoch [ 91/100] | Time: 50.7s | LR: 0.002709 | Train Loss: 0.9425 | Train Acc: 97.10% | Test Loss: 1.5414 | Test Acc: 77.80% <- Best!

✓ Target accuracy 73.0% achieved at epoch 91!
Epoch [ 92/100] | Time: 51.1s | LR: 0.002198 | Train Loss: 0.9321 | Train Acc: 97.48% | Test Loss: 1.5358 | Test Acc: 78.24% <- Best!

✓ Target accuracy 73.0% achieved at epoch 92!
Epoch [ 93/100] | Time: 49.1s | LR: 0.001740 | Train Loss: 0.9308 | Train Acc: 97.47% | Test Loss: 1.5420 | Test Acc: 77.91%

✓ Target accuracy 73.0% achieved at epoch 93!
Epoch [ 94/100] | Time: 49.4s | LR: 0.001334 | Train Loss: 0.9211 | Train Acc: 97.82% | Test Loss: 1.5455 | Test Acc: 77.95%

✓ Target accuracy 73.0% achieved at epoch 94!
Epoch [ 95/100] | Time: 49.5s | LR: 0.000981 | Train Loss: 0.9197 | Train Acc: 97.80% | Test Loss: 1.5382 | Test Acc: 78.02%

✓ Target accuracy 73.0% achieved at epoch 95!
Epoch [ 96/100] | Time: 51.3s | LR: 0.000682 | Train Loss: 0.9159 | Train Acc: 97.95% | Test Loss: 1.5386 | Test Acc: 78.33% <- Best!

✓ Target accuracy 73.0% achieved at epoch 96!
Epoch [ 97/100] | Time: 51.5s | LR: 0.000437 | Train Loss: 0.9137 | Train Acc: 97.98% | Test Loss: 1.5438 | Test Acc: 78.10%

✓ Target accuracy 73.0% achieved at epoch 97!
Epoch [ 98/100] | Time: 51.0s | LR: 0.000246 | Train Loss: 0.9136 | Train Acc: 97.86% | Test Loss: 1.5397 | Test Acc: 78.19%

✓ Target accuracy 73.0% achieved at epoch 98!
Epoch [ 99/100] | Time: 50.7s | LR: 0.000109 | Train Loss: 0.9111 | Train Acc: 98.09% | Test Loss: 1.5385 | Test Acc: 78.11%

✓ Target accuracy 73.0% achieved at epoch 99!
Epoch [100/100] | Time: 50.2s | LR: 0.000027 | Train Loss: 0.9109 | Train Acc: 98.10% | Test Loss: 1.5420 | Test Acc: 78.29%

✓ Target accuracy 73.0% achieved at epoch 100!

================================================================================
TRAINING COMPLETED
================================================================================
Total Training Time: 87.13 minutes (1.45 hours)
Best Test Accuracy: 78.33% (Epoch 96)
Final Test Accuracy: 78.29%
Target Achieved (73.0%): ✓ Yes
================================================================================
