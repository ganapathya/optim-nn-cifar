# CIFAR-100 ResNet-18 Training - Quick Start Guide

This guide will help you quickly train a ResNet-18 model on CIFAR-100 to achieve 73%+ accuracy.

## ⚡ Quick Start (3 Steps)

### Step 1: Verify Setup

```bash
# Test model architecture
cd cifar100
python model.py
```

Expected output: ResNet-18 with ~11M parameters

### Step 2: Start Training

```bash
# Start training (this will take 2-3 hours)
python train.py
```

The training will:

- Run for 100 epochs
- Save checkpoints every 10 epochs
- Save best model automatically
- Log all metrics to `logs/cifar100/`

### Step 3: Monitor Progress

Training logs are saved in real-time to:

```
logs/cifar100/training_log_YYYYMMDD_HHMMSS.txt
```

You can monitor with:

```bash
# Watch training progress
tail -f logs/cifar100/training_log_*.txt
```

## 📊 What to Expect

**Training Timeline (100 epochs):**

- Epoch 1-10: ~30-40% accuracy (learning basics)
- Epoch 20-40: ~55-65% accuracy (rapid improvement)
- Epoch 50-80: ~68-72% accuracy (refinement)
- Epoch 90-100: ~73-75% accuracy (target reached!)

**Time Required:**

- Apple Silicon (M1/M2/M3): ~2-3 hours
- NVIDIA GPU: ~1-2 hours
- CPU: ~12-15 hours (not recommended)

## 🎯 After Training

### Check Results

```bash
# View training summary
cat logs/cifar100/training_log_*.txt | grep "TRAINING COMPLETED" -A 10
```

### Test Inference

```bash
cd cifar100
python inference.py logs/cifar100/best_model.pth
```

### Launch Gradio App

```bash
cd ../gradio_app
export MODEL_PATH=../logs/cifar100/best_model.pth
python app.py
```

Open browser to `http://localhost:7860`

## 🔧 Configuration

All hyperparameters are in `cifar100/config.py`:

```python
# Change batch size if needed
BATCH_SIZE = 128  # Reduce to 64 if OOM

# Change number of epochs
NUM_EPOCHS = 100  # Can reduce for faster testing

# Learning rate
INITIAL_LR = 0.1  # Usually best to keep default
```

## 🐛 Troubleshooting

### "Out of Memory" Error

```python
# In cifar100/config.py
BATCH_SIZE = 64  # or even 32
NUM_WORKERS = 0
```

### "MPS not available"

Training will automatically fall back to CPU. Everything works the same, just slower.

### Training seems stuck

This is normal! Each epoch takes 1-2 minutes. Wait patiently.

### Accuracy too low

- Make sure you train for full 100 epochs
- Check that augmentations are enabled
- Verify learning rate schedule is working

## 📈 Monitoring Tips

### Check Current Best Accuracy

```bash
grep "Best!" logs/cifar100/training_log_*.txt | tail -1
```

### Check Training Speed

```bash
grep "Time:" logs/cifar100/training_log_*.txt | tail -5
```

### Check if Target Reached

```bash
grep "Target Achieved" logs/cifar100/training_log_*.txt
```

## 🚀 Deploy to Huggingface Spaces

After training is complete and you've achieved 73%+ accuracy:

1. Copy trained model to gradio_app:

```bash
cp logs/cifar100/best_model.pth gradio_app/
```

2. Test locally:

```bash
cd gradio_app
python app.py
```

3. Deploy to Huggingface Spaces:

- Create a new Space on Huggingface
- Choose "Gradio" as SDK
- Upload files from `gradio_app/` folder
- Make sure `best_model.pth` is included
- Set Space to public

4. Your app will be live at:

```
https://huggingface.co/spaces/YOUR_USERNAME/cifar100-classifier
```

## 💡 Tips for Best Results

1. **Don't stop early**: The model needs all 100 epochs to reach 73%+
2. **Use GPU**: MPS or CUDA makes training much faster
3. **Monitor logs**: Check progress regularly to ensure training is working
4. **Save checkpoints**: They're saved automatically every 10 epochs
5. **Be patient**: 2-3 hours is normal for quality results

## 📝 Example Training Session

```bash
# Terminal 1: Start training
cd cifar100
python train.py

# Terminal 2: Monitor progress (in another terminal)
cd /path/to/optim-nn-sifar
tail -f logs/cifar100/training_log_*.txt

# Wait 2-3 hours...

# Check final results
grep "Best Test Accuracy" logs/cifar100/training_log_*.txt
# Should see: Best Test Accuracy: 73.XX%

# Test inference
python inference.py logs/cifar100/best_model.pth

# Launch Gradio app
cd ../gradio_app
export MODEL_PATH=../logs/cifar100/best_model.pth
python app.py
```

## ✅ Success Checklist

- [ ] Model architecture verified (`python model.py`)
- [ ] Data loaders tested (`python utils.py`)
- [ ] Training started (`python train.py`)
- [ ] Training completed (100 epochs)
- [ ] Accuracy ≥ 73% achieved
- [ ] Best model saved to `logs/cifar100/best_model.pth`
- [ ] Inference tested and working
- [ ] Gradio app running locally
- [ ] Ready for Huggingface deployment

---

**Need Help?**

- Check `cifar100/README.md` for detailed documentation
- Review training logs in `logs/cifar100/`
- Verify configuration in `cifar100/config.py`

**Happy Training! 🎉**
