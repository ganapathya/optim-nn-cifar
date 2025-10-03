import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CIFAR10Dataset:
    """CIFAR-10 Dataset wrapper with albumentations transforms"""
    
    # CIFAR-10 mean and std for normalization
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    
    def __init__(self, root='./data', train=True, download=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.train = train
        
        if train:
            # Training augmentations using albumentations
            self.transform = A.Compose([
                # 1. Horizontal Flip (as required)
                A.HorizontalFlip(p=0.5),
                
                # 2. ShiftScaleRotate (as required)
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5
                ),
                
                # 3. CoarseDropout (as required with specific parameters)
                # num_holes=1, hole_height=16px, hole_width=16px
                # Updated to use new albumentations 2.0 parameter names
                # Note: In albumentations 2.0+, fill is controlled separately
                A.CoarseDropout(
                    num_holes_range=(1, 1),  # Exactly 1 hole (min_holes=1, max_holes=1)
                    hole_height_range=(16, 16),  # Height = 16px
                    hole_width_range=(16, 16),  # Width = 16px
                    fill=tuple([int(x * 255) for x in self.MEAN]),  # Use 'fill' instead of 'fill_value'
                    p=0.5
                ),
                
                # Normalization
                A.Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ])
        else:
            # Validation/Test transforms (only normalization)
            self.transform = A.Compose([
                A.Normalize(mean=self.MEAN, std=self.STD),
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


def get_dataloaders(batch_size=128, num_workers=4, root='./data'):
    """
    Create train and test dataloaders with albumentations transforms
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        root: Root directory for dataset
    
    Returns:
        train_loader, test_loader
    """
    import torch
    
    # Create datasets
    train_dataset = CIFAR10Dataset(root=root, train=True, download=True)
    test_dataset = CIFAR10Dataset(root=root, train=False, download=True)
    
    # pin_memory should be False for MPS (Apple Silicon)
    # It's only beneficial for CUDA
    use_pin_memory = False  # MPS doesn't support pin_memory well
    
    # Create dataloaders optimized for Apple Silicon
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
    """Print detailed information about albumentations transforms being used"""
    print("\n" + "="*80)
    print("ALBUMENTATIONS TRANSFORMS")
    print("="*80)
    print("\n1. HORIZONTAL FLIP")
    print("   - Probability: 0.5")
    print("   - Randomly flips images horizontally")
    
    print("\n2. SHIFT SCALE ROTATE")
    print("   - Shift limit: ±10%")
    print("   - Scale limit: ±10%")
    print("   - Rotate limit: ±15°")
    print("   - Probability: 0.5")
    
    print("\n3. COARSE DROPOUT (Cutout)")
    print("   - Number of holes: 1 (exactly)")
    print("   - Hole height: 16px")
    print("   - Hole width: 16px")
    print(f"   - Fill value: {CIFAR10Dataset.MEAN} (dataset mean)")
    print("   - Probability: 0.5")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test data loaders
    print("Testing CIFAR-10 data loaders with albumentations...")
    
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
    
    print("\n✓ Data loaders verified successfully!")

