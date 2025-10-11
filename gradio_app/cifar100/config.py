"""
Configuration file for CIFAR-100 ResNet-18 Training
"""

# Model Configuration
MODEL_NAME = "ResNet-18"
NUM_CLASSES = 100

# Data Configuration
BATCH_SIZE = 128
NUM_WORKERS = 0  # Set to 0 for MPS stability, can increase on other devices
DATA_ROOT = './data'

# CIFAR-100 normalization values
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Training Configuration
NUM_EPOCHS = 100
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

# Learning Rate Schedule
LR_WARMUP_EPOCHS = 5
LR_SCHEDULE = 'cosine'  # 'cosine' or 'multistep'

# Data Augmentation (for training)
RANDOM_CROP_PADDING = 4
RANDOM_HORIZONTAL_FLIP = True
CUTOUT_SIZE = 16
CUTOUT_PROB = 0.5

# Checkpointing
SAVE_CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs
SAVE_BEST_MODEL = True

# Device Configuration
USE_MPS = True  # Use Apple Silicon GPU if available

# Logging
LOG_DIR = 'logs/cifar100'
VERBOSE = True

# Target Accuracy
TARGET_ACCURACY = 73.0

