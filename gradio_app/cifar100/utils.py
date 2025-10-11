"""
Data utilities for CIFAR-100
Includes dataset loaders with albumentations augmentations
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    BATCH_SIZE, NUM_WORKERS, DATA_ROOT,
    CIFAR100_MEAN, CIFAR100_STD,
    RANDOM_CROP_PADDING, CUTOUT_SIZE, CUTOUT_PROB
)


# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class CIFAR100Dataset:
    """CIFAR-100 Dataset wrapper with albumentations transforms"""
    
    def __init__(self, root=DATA_ROOT, train=True, download=True):
        self.dataset = datasets.CIFAR100(root=root, train=train, download=download)
        self.train = train
        
        if train:
            # Training augmentations using albumentations
            self.transform = A.Compose([
                # RandomCrop with padding
                A.PadIfNeeded(min_height=32+RANDOM_CROP_PADDING*2, 
                             min_width=32+RANDOM_CROP_PADDING*2, 
                             border_mode=0, p=1.0),
                A.RandomCrop(height=32, width=32, p=1.0),
                
                # Horizontal Flip
                A.HorizontalFlip(p=0.5),
                
                # Cutout (CoarseDropout)
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(CUTOUT_SIZE, CUTOUT_SIZE),
                    hole_width_range=(CUTOUT_SIZE, CUTOUT_SIZE),
                    fill=tuple([int(x * 255) for x in CIFAR100_MEAN]),
                    p=CUTOUT_PROB
                ),
                
                # Normalization
                A.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
                ToTensorV2(),
            ])
        else:
            # Test transforms (only normalization)
            self.transform = A.Compose([
                A.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        # Apply albumentations transforms
        augmented = self.transform(image=image)
        image = augmented['image']
        
        return image, label


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, root=DATA_ROOT):
    """
    Create train and test dataloaders for CIFAR-100
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        root: Root directory for dataset
    
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = CIFAR100Dataset(root=root, train=True, download=True)
    test_dataset = CIFAR100Dataset(root=root, train=False, download=True)
    
    # pin_memory should be False for MPS (Apple Silicon)
    use_pin_memory = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


def print_augmentation_info():
    """Print detailed information about augmentations being used"""
    print("\n" + "="*80)
    print("CIFAR-100 DATA AUGMENTATION")
    print("="*80)
    print("\n1. RANDOM CROP")
    print(f"   - Padding: {RANDOM_CROP_PADDING} pixels")
    print(f"   - Output size: 32x32")
    
    print("\n2. HORIZONTAL FLIP")
    print("   - Probability: 0.5")
    
    print("\n3. CUTOUT (CoarseDropout)")
    print(f"   - Number of holes: 1")
    print(f"   - Hole size: {CUTOUT_SIZE}x{CUTOUT_SIZE} pixels")
    print(f"   - Probability: {CUTOUT_PROB}")
    print(f"   - Fill value: {CIFAR100_MEAN} (dataset mean)")
    
    print("\n4. NORMALIZATION")
    print(f"   - Mean: {CIFAR100_MEAN}")
    print(f"   - Std: {CIFAR100_STD}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test data loaders
    print("Testing CIFAR-100 data loaders with albumentations...")
    
    print_augmentation_info()
    
    train_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of batches (train): {len(train_loader)}")
    print(f"Number of batches (test): {len(test_loader)}")
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Sample labels: {labels.tolist()}")
    print(f"Sample classes: {[CIFAR100_CLASSES[i] for i in labels.tolist()]}")
    
    print("\n✓ Data loaders verified successfully!")

