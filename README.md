# Neural Network Training Projects

This repository contains implementations for training deep learning models on CIFAR datasets with optimized architectures and training strategies.

## 📁 Project Structure

```
optim-nn-sifar/
├── cifar10/              # CIFAR-10 custom CNN (88.59% accuracy)
│   ├── model.py          # Custom efficient CNN architecture
│   ├── train.py          # Training pipeline
│   ├── utils.py          # Data loaders with augmentations
│   └── test_setup.py     # Setup verification
│
├── cifar100/             # CIFAR-100 ResNet-18 (Target: 73%+)
│   ├── model.py          # ResNet-18 adapted for CIFAR-100
│   ├── train.py          # Training with cosine annealing
│   ├── utils.py          # Data loaders and augmentations
│   ├── inference.py      # Model inference utilities
│   ├── config.py         # Training configuration
│   └── README.md         # Detailed documentation
│
├── gradio_app/           # Huggingface Spaces deployment
│   ├── app.py            # Gradio interface for CIFAR-100
│   ├── requirements.txt  # Deployment dependencies
│   └── README.md         # App documentation
│
├── data/                 # Datasets (auto-downloaded)
├── logs/                 # Training logs and checkpoints
│   ├── cifar100/         # CIFAR-100 training logs
│   └── ...
│
├── requirements.txt      # Main project dependencies
└── README.md            # This file
```

## 🎯 Projects

### 1. CIFAR-10 Custom CNN ✅ Completed

A highly efficient CNN achieving **88.59% accuracy** with only **183,802 parameters** (<200k).

**Key Features:**

- Custom C1C2C3C40 architecture
- Depthwise separable convolutions
- Dilated convolutions (no MaxPooling)
- Global Average Pooling
- Receptive field: 47
- Training time: ~14 minutes on Apple Silicon

**Status:** ✅ Completed and exceeds 85% target

📖 **See:** `cifar10/` folder for implementation

---

### 2. CIFAR-100 ResNet-18 🔄 Ready for Training

ResNet-18 model trained from scratch on CIFAR-100, targeting **73%+ top-1 accuracy**.

**Architecture:**

- ResNet-18 adapted for 32×32 images
- ~11 million parameters
- Modified first conv (3×3 instead of 7×7)
- No initial maxpool

**Training Strategy:**

- 100 epochs with cosine annealing
- SGD with momentum (0.9) and weight decay (5e-4)
- Label smoothing (0.1)
- Strong augmentations: RandomCrop, HFlip, Cutout
- Warmup for first 5 epochs

**Expected Training Time:** ~2-3 hours on Apple Silicon

📖 **See:** `cifar100/README.md` for detailed guide

---

### 3. Huggingface Gradio App 🌐

Interactive web application for CIFAR-100 image classification, ready for deployment on Huggingface Spaces.

**Features:**

- Upload image for classification
- Top-5 predictions with confidence scores
- Clean, modern UI with Gradio
- Model information and documentation

📖 **See:** `gradio_app/README.md` for deployment guide

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd optim-nn-sifar

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### CIFAR-10 (Already Trained)

The CIFAR-10 model is already trained with saved checkpoints in `logs/`.

```bash
# View model architecture
cd cifar10
python model.py

# View training results
cat ../logs/training_log_*.txt
```

### CIFAR-100 (Train New Model)

```bash
# Navigate to cifar100 directory
cd cifar100

# Test model architecture
python model.py

# Test data loaders
python utils.py

# Start training (100 epochs, ~2-3 hours)
python train.py

# After training, test inference
python inference.py logs/cifar100/best_model.pth
```

### Launch Gradio App (After Training)

```bash
# From project root
cd gradio_app

# Make sure model is trained and available
export MODEL_PATH=../logs/cifar100/best_model.pth

# Launch app
python app.py
```

The app will be available at `http://localhost:7860`

---

## 🎓 Key Techniques & Technologies

### Deep Learning

- **ResNet Architecture** - Residual learning with skip connections
- **Label Smoothing** - Prevents overconfident predictions
- **Cosine Annealing** - Smooth learning rate decay
- **Warmup Scheduling** - Gradual LR increase at start
- **Depthwise Separable Convolutions** - Parameter efficiency
- **Dilated Convolutions** - Expand receptive field
- **Global Average Pooling** - Reduce overfitting

### Data Augmentation

- Random Crop with padding
- Random Horizontal Flip
- Cutout / CoarseDropout
- Albumentations library

### Optimization

- SGD with Nesterov momentum
- Weight decay (L2 regularization)
- Batch normalization
- Gradient clipping (if needed)

### Tools & Frameworks

- **PyTorch** - Deep learning framework
- **Albumentations** - Advanced image augmentation
- **Gradio** - Web interface for ML models
- **Apple Silicon MPS** - GPU acceleration on Mac

---

## 📊 Results Summary

| Project   | Model      | Dataset   | Accuracy          | Parameters | Status            |
| --------- | ---------- | --------- | ----------------- | ---------- | ----------------- |
| CIFAR-10  | Custom CNN | CIFAR-10  | **88.59%**        | 183,802    | ✅ Completed      |
| CIFAR-100 | ResNet-18  | CIFAR-100 | **73%+ (target)** | ~11M       | 🔄 Ready to train |

---

## 💻 System Requirements

### Hardware

- **Recommended:** Apple Silicon (M1/M2/M3) or NVIDIA GPU
- **Minimum:** CPU (slower training)
- **RAM:** 8GB+ recommended
- **Storage:** ~2GB for datasets and models

### Software

- **OS:** macOS 12.3+ (for MPS), Linux, or Windows
- **Python:** 3.8+
- **PyTorch:** 2.0+ with MPS or CUDA support

---

## 📈 Training Progress Tracking

### CIFAR-100 Training Logs

Logs are saved to `logs/cifar100/` with timestamps:

```
logs/cifar100/
├── training_log_YYYYMMDD_HHMMSS.txt  # Complete training log
├── best_model.pth                     # Best checkpoint
├── checkpoint_epoch_10.pth            # Periodic checkpoints
├── checkpoint_epoch_20.pth
└── ...
```

Each log includes:

- Epoch-by-epoch training and validation metrics
- Learning rate schedule
- Best accuracy tracking
- Training time per epoch
- Final summary with total time and best results

---

## 🔧 Configuration

### CIFAR-100 Training Config

All hyperparameters are centralized in `cifar100/config.py`:

```python
# Model
NUM_CLASSES = 100

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 100
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

# Schedule
LR_WARMUP_EPOCHS = 5
LR_SCHEDULE = 'cosine'

# Augmentation
RANDOM_CROP_PADDING = 4
CUTOUT_SIZE = 16
CUTOUT_PROB = 0.5
```

Easy to modify for experimentation!

---

## 🐛 Troubleshooting

### MPS (Apple Silicon) Issues

If MPS is not available:

```python
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Fallback to CPU in config.py
USE_MPS = False
```

### Memory Issues

Reduce batch size in `config.py`:

```python
BATCH_SIZE = 64  # or 32
NUM_WORKERS = 0  # reduce workers
```

### Slow Training

- Ensure GPU (MPS/CUDA) is being used
- Close other applications
- Check that data is not being downloaded repeatedly

---

## 📚 References

### Papers

1. He et al., "Deep Residual Learning for Image Recognition" (2016)
2. Krizhevsky & Hinton, "Learning Multiple Layers of Features from Tiny Images" (2009)
3. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (2016)
4. Smith, "A disciplined approach to neural network hyper-parameters" (2018)

### Datasets

- **CIFAR-10:** 60,000 32×32 images, 10 classes
- **CIFAR-100:** 60,000 32×32 images, 100 classes

---

## 🤝 Contributing

Feel free to:

- Report issues
- Suggest improvements
- Submit pull requests
- Share training results

---

## 📝 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- PyTorch team for excellent framework
- CIFAR dataset creators
- Albumentations library contributors
- Gradio team for easy ML deployment
- Apple for MPS support in PyTorch

---

## 🎯 Next Steps

1. ✅ CIFAR-10 custom CNN trained and validated
2. 🔄 Train CIFAR-100 ResNet-18 model
3. 📊 Validate 73%+ accuracy target
4. 🚀 Deploy to Huggingface Spaces
5. 🌐 Share the live application

---

**Happy Training! 🚀**

For detailed instructions on each project, see the respective README files in each folder.
