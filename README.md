# CIFAR-10 CNN - Apple Silicon Optimized

A highly efficient Convolutional Neural Network for CIFAR-10 classification, optimized for Apple Silicon GPUs (MPS). This implementation achieves **88.59% accuracy** (exceeding the 85% target) with only **183,802 parameters** (<200k) using modern CNN techniques.

## 🎯 Project Results

| Objective                | Target                   | Achieved                        | Status           |
| ------------------------ | ------------------------ | ------------------------------- | ---------------- |
| Architecture             | C1C2C3C40                | C1C2C3C40                       | ✅               |
| No MaxPooling            | Use dilated convolutions | Dilated convolutions used       | ✅               |
| Receptive Field          | > 44                     | 47                              | ✅               |
| Depthwise Separable Conv | Required                 | Implemented in C2               | ✅               |
| Dilated Convolution      | Required                 | Implemented in C3 & transitions | ✅               |
| Global Average Pooling   | Required                 | Implemented                     | ✅               |
| Albumentations           | 3 transforms             | All 3 implemented               | ✅               |
| **Target Accuracy**      | **85%**                  | **88.59%**                      | ✅ **Exceeded!** |
| Parameters               | < 200k                   | 183,802                         | ✅               |
| Apple Silicon MPS        | Optimized                | Fully optimized                 | ✅               |
| **Training Time**        | ~15-20 min               | **13.88 min**                   | ✅ **Faster!**   |

---

## 📊 Model Architecture

### Network Structure: C1C2C3C40

The network follows a C1C2C3C40 architecture with the following characteristics:

```
Input (3x32x32)
    ↓
┌─────────────────────────────────────────┐
│ C1 Block                                │
│ - Conv 3→20 (3x3, pad=1)        RF: 3  │
│ - Conv 20→28 (3x3, pad=1)       RF: 5  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Transition 1 (Dilated Conv)             │
│ - Conv 28→28 (3x3, s=2, d=1)    RF: 7  │
│   Output: 16x16                         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ C2 Block (Depthwise Separable)          │
│ - DepthwiseSeparable 28→40      RF: 11 │
│ - Conv 40→48 (3x3, pad=1)       RF: 15 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Transition 2 (Dilated Conv)             │
│ - Conv 48→48 (3x3, s=2, d=2)    RF: 23 │
│   Output: 8x8                           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ C3 Block (Dilated Convolution)          │
│ - Conv 48→56 (3x3, d=2)         RF: 31 │
│ - Conv 56→64 (3x3, pad=1)       RF: 35 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ C40 Block                               │
│ - Conv 64→72 (3x3, pad=1)       RF: 39 │
│ - Conv 72→56 (1x1)              RF: 39 │
│ - Conv 56→56 (3x3, s=2)         RF: 47 │
│   Output: 4x4                           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Global Average Pooling (GAP)            │
│   Output: 1x1                           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Fully Connected Layer                   │
│ - FC 56→10                              │
└─────────────────────────────────────────┘
    ↓
Output (10 classes)
```

### Key Features

1. **Depthwise Separable Convolution** (C2 Block)

   - Reduces parameters while maintaining performance
   - Separates spatial and channel-wise operations

2. **Dilated Convolution** (C3 Block & Transitions)

   - Increases receptive field without increasing parameters
   - Replaces MaxPooling for downsampling
   - Dilation rates: 1, 2

3. **Global Average Pooling (GAP)**

   - Reduces overfitting compared to FC layers
   - Minimizes parameters in final layers

4. **Final Receptive Field: 47** (> 44 requirement ✓)

### Parameter Count

```
Total Parameters: 183,802 (< 200k ✓)
Trainable Parameters: 183,802
```

---

## 🔄 Data Augmentation (Albumentations)

All three required augmentations are implemented in `utils.py`:

### 1. Horizontal Flip

```python
A.HorizontalFlip(p=0.5)
```

- Randomly flips images horizontally
- Probability: 50%

### 2. ShiftScaleRotate

```python
A.ShiftScaleRotate(
    shift_limit=0.1,
    scale_limit=0.1,
    rotate_limit=15,
    border_mode=0,
    p=0.5
)
```

- Shift: ±10%
- Scale: ±10%
- Rotation: ±15°
- Probability: 50%

### 3. CoarseDropout (Cutout)

```python
A.CoarseDropout(
    max_holes=1,
    max_height=16,
    max_width=16,
    min_holes=1,
    min_height=16,
    min_width=16,
    fill_value=(125, 123, 114),  # CIFAR-10 mean
    mask_fill_value=None,
    p=0.5
)
```

- Creates exactly 1 hole of 16x16 pixels
- Filled with dataset mean values
- Probability: 50%

---

## 🚀 Installation & Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- PyTorch with MPS support

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd optim-nn-sifar

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. View Model Summary

```bash
python model.py
```

This will display:

- Layer-wise architecture
- Parameter count per layer
- Total parameters
- Receptive field verification

### 2. Test Data Loaders & Augmentations

```bash
python utils.py
```

This will show:

- Albumentations transform details
- Dataset sizes
- Sample batch information

### 3. Train the Model

```bash
python train.py
```

Training configuration:

- **Epochs**: 50 (adjustable)
- **Batch Size**: 128
- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Learning Rate**: OneCycleLR (max_lr=0.1)
- **Weight Decay**: 1e-4
- **Device**: Apple Silicon MPS (automatic detection)

### 4. Monitor Training

Training logs are saved in `logs/` directory:

- `training_log_<timestamp>.txt` - Complete training log
- `best_model.pth` - Best model checkpoint
- `checkpoint_epoch_<N>.pth` - Periodic checkpoints

---

## 📁 Project Structure

```
optim-nn-sifar/
│
├── model.py              # CNN architecture (C1C2C3C40)
├── utils.py              # Data loaders & augmentations
├── train.py              # Training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
├── data/                # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
│
└── logs/                # Training logs & checkpoints
    ├── training_log_<timestamp>.txt
    ├── best_model.pth
    └── checkpoint_epoch_<N>.pth
```

---

## 🎓 Training Results

### ✅ Achieved Performance (Apple Silicon M-series)

| Metric             | Target       | Achieved          | Status                    |
| ------------------ | ------------ | ----------------- | ------------------------- |
| **Test Accuracy**  | 85%+         | **88.59%**        | ✅ **Exceeded by 3.59%!** |
| **Training Time**  | ~15-20 min   | **13.88 minutes** | ✅ Faster than expected   |
| **Convergence**    | 40-50 epochs | 50 epochs         | ✅                        |
| **Target Reached** | -            | Epoch 40 (86.62%) | ✅ 85%+ by epoch 40       |
| **Parameters**     | < 200k       | 183,802           | ✅                        |

### Training Progression

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc   | Notes                |
| ----- | ---------- | --------- | --------- | ---------- | -------------------- |
| 1     | 1.7333     | 35.46%    | 1.3756    | 49.78%     | Initial baseline     |
| 10    | 0.7775     | 73.18%    | 0.6719    | 76.81%     | Rapid learning       |
| 20    | 0.6379     | 77.83%    | 0.4910    | 83.10%     | Approaching target   |
| 30    | 0.5613     | 80.52%    | 0.4512    | 84.77%     | Near 85%             |
| 40    | 0.4782     | 83.35%    | 0.3938    | 86.62%     | **Target exceeded!** |
| 50    | 0.3902     | 86.51%    | 0.3412    | **88.59%** | **Final best**       |

### Key Observations

- ✅ **Consistent improvement** throughout all 50 epochs
- ✅ **No overfitting** - Train accuracy (86.51%) vs Test accuracy (88.59%)
- ✅ **Stable training** on Apple Silicon MPS with no device errors
- ✅ **Fast convergence** - 85% target reached by epoch 40
- ✅ **Efficient architecture** - 183k parameters achieving 88.59%

### Actual Training Log Sample

```
Epoch [  1/50] | Time: 11.8s | LR: 0.006350 | Train Loss: 1.7333 | Train Acc: 35.46% | Test Loss: 1.3756 | Test Acc: 49.78% <- Best!
Epoch [ 10/50] | Time: 19.5s | LR: 0.100000 | Train Loss: 0.7775 | Train Acc: 73.18% | Test Loss: 0.6719 | Test Acc: 76.81%
Epoch [ 20/50] | Time: 10.8s | LR: 0.085348 | Train Loss: 0.6379 | Train Acc: 77.83% | Test Loss: 0.4910 | Test Acc: 83.10% <- Best!
Epoch [ 30/50] | Time: 19.3s | LR: 0.049990 | Train Loss: 0.5613 | Train Acc: 80.52% | Test Loss: 0.4512 | Test Acc: 84.77% <- Best!
Epoch [ 40/50] | Time: 19.2s | LR: 0.014638 | Train Loss: 0.4782 | Train Acc: 83.35% | Test Loss: 0.3938 | Test Acc: 86.62% <- Best!
Epoch [ 50/50] | Time: 13.2s | LR: 0.000000 | Train Loss: 0.3902 | Train Acc: 86.51% | Test Loss: 0.3412 | Test Acc: 88.59% <- Best!

================================================================================
TRAINING COMPLETED
================================================================================
Total Training Time: 13.88 minutes
Best Test Accuracy: 88.59% (Epoch 50)
Target Achieved (85%): ✓ Yes
================================================================================
```

### Validation After Each Epoch

✅ The training script automatically runs validation after each epoch and logs:

- Training loss & accuracy
- Test loss & accuracy
- Learning rate
- Best model tracking
- Automatic checkpointing

---

## 🔧 Apple Silicon Optimization

### MPS (Metal Performance Shaders) Features

1. **Automatic Device Selection**

   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

2. **Optimized Data Loading**

   - `pin_memory=True` for faster GPU transfer
   - `persistent_workers=True` for worker reuse
   - `num_workers=4` optimized for Apple Silicon

3. **Efficient Memory Usage**
   - Batch size tuned for M-series GPUs
   - Gradient accumulation support

### Performance Tips

- **M1/M2 Macs**: Use batch_size=128
- **Limited RAM**: Reduce batch_size to 64 or set num_workers=2
- **Memory Issues**: Close other applications during training

---

## 📊 Model Specifications

| Specification      | Value              | Status      |
| ------------------ | ------------------ | ----------- |
| Architecture       | C1C2C3C40          | ✅          |
| Receptive Field    | 47                 | ✅ (> 44)   |
| Total Parameters   | ~180k              | ✅ (< 200k) |
| Depthwise Sep Conv | C2 Block           | ✅          |
| Dilated Conv       | C3 + Transitions   | ✅          |
| GAP                | Yes                | ✅          |
| Target Accuracy    | 85%                | ✅          |
| Augmentations      | 3 (Albumentations) | ✅          |
| Apple Silicon      | MPS Optimized      | ✅          |

---

## 🧪 Testing & Validation

### Verify Model Architecture

```python
from model import CIFAR10Net, get_model_summary
import torch

model = CIFAR10Net()
device = torch.device("mps")
get_model_summary(model, device=device)
```

### Verify Augmentations

```python
from utils import print_augmentation_info
print_augmentation_info()
```

### Load & Test Best Model

```python
import torch
from model import CIFAR10Net

model = CIFAR10Net()
checkpoint = torch.load('logs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best Accuracy: {checkpoint['accuracy']:.2f}%")
print(f"Epoch: {checkpoint['epoch']}")
```

---

## 📚 Key Technologies

- **PyTorch**: Deep learning framework
- **Albumentations**: Advanced image augmentation
- **torchsummary**: Model architecture visualization
- **tqdm**: Progress bars for training

---

## 🔍 Architecture Highlights

### Why This Design Works

1. **Dilated Convolutions over MaxPooling**

   - Maintains spatial resolution longer
   - Increases receptive field without parameter cost
   - Better gradient flow

2. **Depthwise Separable Convolutions**

   - 8-9x fewer parameters than standard convolutions
   - Nearly same representational power
   - Faster training and inference

3. **Global Average Pooling**

   - Acts as structural regularizer
   - Reduces parameters significantly
   - More robust to spatial translations

4. **OneCycleLR Scheduler**
   - Faster convergence
   - Better generalization
   - Automatic learning rate tuning

---

## 🐛 Troubleshooting

### MPS Not Available

If you see "MPS not available, using CPU":

- Ensure you have macOS 12.3+ and Apple Silicon
- Update PyTorch: `pip install --upgrade torch torchvision`
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory

If you encounter OOM errors:

```python
# In train.py, reduce batch size
train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=2)
```

### Slow Training

- Close other applications
- Reduce `num_workers` to 2
- Ensure no background processes are using GPU

---

## 🏆 Final Summary

### Achievement Highlights

This CIFAR-10 CNN project successfully demonstrates:

✅ **Exceeded Target Accuracy**: Achieved **88.59%** (target was 85%+) - **3.59% above target**

✅ **Efficient Architecture**: Only **183,802 parameters** (< 200k requirement)

- Depthwise Separable Convolution reduces params while maintaining performance
- Dilated Convolutions replace MaxPooling for better receptive field
- Global Average Pooling minimizes final layer parameters

✅ **Fast Training**: **13.88 minutes** on Apple Silicon (faster than 15-20 min target)

- Fully optimized for Apple Silicon MPS
- No device mismatch errors
- Stable training with OneCycleLR scheduler

✅ **Modern Techniques**:

- C1C2C3C40 architecture with RF=47 (> 44 requirement)
- 3 Albumentations transforms (HorizontalFlip, ShiftScaleRotate, CoarseDropout)
- No MaxPooling - uses dilated convolutions instead
- Validation after each epoch with automatic checkpointing

✅ **Clean Code**:

- Modular design (`model.py`, `utils.py`, `train.py`)
- Comprehensive documentation
- Easy to run and reproduce

### Performance Comparison

| Metric          | Expected  | Actual    | Improvement    |
| --------------- | --------- | --------- | -------------- |
| Accuracy        | 85%       | 88.59%    | +3.59%         |
| Training Time   | 15-20 min | 13.88 min | ~25% faster    |
| Parameters      | < 200k    | 183,802   | 8% under limit |
| Receptive Field | > 44      | 47        | Exceeded       |

### Why It Works

1. **Dilated Convolutions**: Increase receptive field without increasing parameters
2. **Depthwise Separable**: ~8x fewer parameters than standard convolutions
3. **GAP**: Reduces overfitting and parameters in final layers
4. **OneCycleLR**: Super-convergence for faster, better training
5. **Albumentations**: Strong data augmentation prevents overfitting
6. **Apple Silicon MPS**: Optimized for unified memory architecture

**All requirements met and exceeded!** 🎉

---

## 📝 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- Albumentations library for augmentations
- PyTorch team for MPS support

---

## 📧 Contact

For questions or improvements, please open an issue in the repository.

---

**Happy Training! 🚀**
