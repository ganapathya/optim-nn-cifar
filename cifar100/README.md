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

**Happy Training! 🎉**
