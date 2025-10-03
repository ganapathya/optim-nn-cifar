# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup (30 seconds)

```bash
python test_setup.py
```

Expected output:

```
✓ ALL TESTS PASSED - READY FOR TRAINING!
```

### Step 3: Start Training (15-20 minutes)

```bash
python train.py
```

That's it! Training will start automatically.

---

## 📊 What to Expect

### During Training

You'll see progress bars like this:

```
Epoch 1: 100%|████████| 391/391 [00:18<00:00, 21.2it/s, Loss=1.523, Acc=44.32%, LR=0.020000]

Epoch [  1/ 50] | Time: 18.2s | LR: 0.020000 |
Train Loss: 1.5234 | Train Acc: 44.32% |
Test Loss: 1.2345 | Test Acc: 55.67%
```

### After Training

Logs and models saved to `logs/`:

- `training_log_<timestamp>.txt` - Full training history
- `best_model.pth` - Best performing checkpoint
- `checkpoint_epoch_<N>.pth` - Periodic saves

---

## 🎯 Key Features

✅ **Architecture:** C1C2C3C40 (4 conv blocks)
✅ **Parameters:** 183,802 (< 200k requirement)
✅ **Receptive Field:** 47 (> 44 requirement)
✅ **Target Accuracy:** 85%+
✅ **Depthwise Separable Conv:** In C2 block
✅ **Dilated Convolution:** In C3 & transitions
✅ **No MaxPooling:** Uses dilated/strided conv
✅ **GAP:** Global Average Pooling
✅ **Augmentations:** 3 albumentations transforms
✅ **Apple Silicon:** MPS optimized

---

## 🧪 Individual Component Tests

### Test Model Only

```bash
python model.py
```

Shows: Architecture summary, parameter count, RF verification

### Test Data Loaders Only

```bash
python utils.py
```

Shows: Augmentation details, dataset info

### Test Everything

```bash
python test_setup.py
```

Shows: Complete verification checklist

---

## 📁 Project Structure

```
optim-nn-sifar/
├── model.py              # CNN architecture
├── utils.py              # Data loaders & augmentations
├── train.py              # Training script
├── test_setup.py         # Verification script
├── requirements.txt      # Dependencies
├── README.md            # Full documentation
├── ARCHITECTURE.md      # Technical details
├── QUICKSTART.md        # This file
└── logs/                # Training outputs (created)
    ├── training_log_*.txt
    ├── best_model.pth
    └── checkpoint_epoch_*.pth
```

---

## ⚙️ Customization

### Change Number of Epochs

Edit `train.py` line 202:

```python
history = trainer.train(num_epochs=50)  # Change 50 to desired
```

### Change Batch Size

Edit `train.py` line 196:

```python
train_loader, test_loader = get_dataloaders(
    batch_size=128,  # Change 128 to desired
    num_workers=4,
    root='./data'
)
```

### Change Learning Rate

Edit `train.py` line 81:

```python
self.scheduler = OneCycleLR(
    self.optimizer,
    max_lr=0.1,  # Change 0.1 to desired
    epochs=num_epochs,
    steps_per_epoch=len(self.train_loader)
)
```

---

## 🐛 Troubleshooting

### MPS Not Available

**Issue:** "MPS not available, using CPU"

**Solution:**

- Ensure macOS 12.3+ and Apple Silicon
- Update PyTorch: `pip install --upgrade torch torchvision`

### Out of Memory

**Issue:** "RuntimeError: MPS backend out of memory"

**Solution:** Reduce batch size in `train.py`:

```python
batch_size=64  # or even 32
```

### Slow Training

**Issue:** Training very slow

**Possible causes:**

1. Close other applications
2. Reduce num_workers to 2
3. Check Activity Monitor for background processes

---

## 📊 Monitoring Training

### Real-time Monitoring

Watch the log file in real-time:

```bash
tail -f logs/training_log_*.txt
```

### Check Best Accuracy

```bash
grep "Best!" logs/training_log_*.txt
```

### Load Best Model

```python
import torch
from model import CIFAR10Net

model = CIFAR10Net()
checkpoint = torch.load('logs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best Accuracy: {checkpoint['accuracy']:.2f}%")
```

---

## 📚 Documentation

- **Quick Start:** `QUICKSTART.md` (this file)
- **Full Guide:** `README.md`
- **Architecture:** `ARCHITECTURE.md`
- **Code:** Inline comments in all `.py` files

---

## 🎓 Learning Resources

### Understanding the Architecture

1. Run `python model.py` to see layer-by-layer breakdown
2. Check `ARCHITECTURE.md` for receptive field calculations
3. Review code comments in `model.py`

### Understanding Augmentations

1. Run `python utils.py` to see transform details
2. Check sample images before/after augmentation
3. Review `utils.py` lines 23-45

---

## ✅ Verification Checklist

Before training, verify:

- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] MPS available (`python -c "import torch; print(torch.backends.mps.is_available())"`)
- [ ] Test passes (`python test_setup.py`)
- [ ] Disk space available (>2GB for dataset + logs)

---

## 🎯 Expected Results

After 50 epochs on Apple Silicon:

- **Best Test Accuracy:** 85%+
- **Training Time:** ~15-20 minutes
- **Final Model Size:** ~0.7 MB
- **Log File Size:** ~50 KB

---

## 🚀 Next Steps After Training

1. **Evaluate model:**

   ```bash
   python -c "from train import *; # Load and evaluate"
   ```

2. **Visualize training:**

   - Plot accuracy curves from training log
   - Analyze loss progression

3. **Experiment:**
   - Try different learning rates
   - Adjust augmentation probabilities
   - Modify architecture channels

---

**Ready to train? Run:** `python train.py`

Good luck! 🎉
