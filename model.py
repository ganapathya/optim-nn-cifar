import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: Depthwise + Pointwise (1x1) convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class CIFAR10Net(nn.Module):
    """
    CIFAR-10 CNN with architecture C1C2C3C40
    - Uses Dilated Convolutions instead of MaxPooling
    - Uses Depthwise Separable Convolution
    - Total RF > 44
    - Params < 200k
    - Optimized for Apple Silicon MPS
    """
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        
        # C1 Block - Initial feature extraction with reduced channels
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 28, kernel_size=3, padding=1, bias=False),  # RF: 5
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
        )
        
        # Transition with dilated convolution (instead of MaxPooling) - reduces spatial dims
        self.trans1 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),  # RF: 7, spatial: 16x16
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
        )
        
        # C2 Block - Using Depthwise Separable Convolution (parameter efficient)
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(28, 40, kernel_size=3, padding=1),  # RF: 11
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 48, kernel_size=3, padding=1, bias=False),  # RF: 15
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        # Transition with dilated convolution (dilation=2) - maintains spatial dims
        self.trans2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),  # RF: 23, spatial: 8x8
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        # C3 Block - Using Dilated Convolution with dilation=2
        self.c3 = nn.Sequential(
            nn.Conv2d(48, 56, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 31 (dilated conv)
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 64, kernel_size=3, padding=1, bias=False),  # RF: 35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # C40 Block - Final feature extraction before GAP
        # Using stride=2 in the last layer as per requirement
        # Using 1x1 convolution for channel reduction (efficient)
        self.c40 = nn.Sequential(
            nn.Conv2d(64, 72, kernel_size=3, padding=1, bias=False),  # RF: 39
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 56, kernel_size=1, bias=False),  # 1x1 conv for channel reduction
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 56, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 47, spatial: 4x4
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling (GAP) - compulsory
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected layer after GAP to target number of classes
        self.fc = nn.Linear(56, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, x):
        x = self.c1(x)           # 32x32
        x = self.trans1(x)       # 16x16
        x = self.c2(x)           # 16x16
        x = self.trans2(x)       # 8x8
        x = self.c3(x)           # 8x8
        x = self.c40(x)          # 4x4
        x = self.gap(x)          # 1x1
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_model_summary(model, input_size=(3, 32, 32), device='cpu'):
    """
    Get model summary with parameter count and layer details
    
    Note: torchsummary doesn't support MPS, so we always run on CPU
    and then move the model back to the original device.
    """
    from torchsummary import summary
    
    # Store original device
    original_device = next(model.parameters()).device
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    
    # Always use CPU for torchsummary (MPS not supported)
    summary(model.to('cpu'), input_size)
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameters < 200k: {'✓ Yes' if total_params < 200000 else '✗ No'}")
    print("="*80 + "\n")
    
    # Move model back to original device
    model.to(original_device)
    
    return total_params


if __name__ == "__main__":
    # Test the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CIFAR10Net(num_classes=10)
    
    # Get model summary - use CPU for torchsummary compatibility
    print("\nGenerating model summary (on CPU for compatibility)...")
    get_model_summary(model, device='cpu')
    
    # Test forward pass on actual device
    print(f"\nTesting forward pass on {device}...")
    x = torch.randn(2, 3, 32, 32).to(device)
    model = model.to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Model architecture verified successfully!")

