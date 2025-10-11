"""
Training script for ResNet-18 on CIFAR-100
Target: 73% top-1 accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import time
import os
from datetime import datetime

from model import resnet18, get_model_summary
from utils import get_dataloaders, print_augmentation_info, CIFAR100_CLASSES
from config import *


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Helps with generalization by preventing overconfident predictions
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class WarmUpLR(LambdaLR):
    """Warmup learning rate scheduler"""
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
    def lr_lambda(self, cur_iter):
        return cur_iter / self.total_iters


class Trainer:
    """Training manager for CIFAR-100 ResNet-18"""
    
    def __init__(self, model, train_loader, test_loader, device, config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        
        # Optimizer - SGD with momentum and weight decay
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=INITIAL_LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True
        )
        
        # Learning rate scheduler
        self.scheduler = None
        self.warmup_scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': []
        }
        
        # Best accuracy tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        
        # Create logs directory
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Log file path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(LOG_DIR, f'training_log_{timestamp}.txt')
        
        # Write header to log file
        self.write_log("="*80)
        self.write_log("CIFAR-100 ResNet-18 TRAINING LOG")
        self.write_log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_log(f"Device: {device}")
        self.write_log(f"Model: {MODEL_NAME}")
        self.write_log(f"Target Accuracy: {TARGET_ACCURACY}%")
        self.write_log("="*80 + "\n")
    
    def write_log(self, message):
        """Write message to both console and log file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update warmup scheduler (if in warmup period)
            if self.warmup_scheduler is not None and epoch <= LR_WARMUP_EPOCHS:
                self.warmup_scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return epoch_loss, epoch_acc, current_lr
    
    def validate(self):
        """Validate on test set"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Validating', leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, num_epochs=NUM_EPOCHS):
        """Main training loop"""
        self.write_log(f"\nStarting training for {num_epochs} epochs...")
        self.write_log(f"Batch size: {BATCH_SIZE}")
        self.write_log(f"Initial learning rate: {INITIAL_LR}")
        self.write_log(f"Weight decay: {WEIGHT_DECAY}")
        self.write_log(f"Label smoothing: {LABEL_SMOOTHING}")
        self.write_log(f"Warmup epochs: {LR_WARMUP_EPOCHS}")
        self.write_log("="*80 + "\n")
        
        # Initialize schedulers
        # Warmup scheduler for first few epochs
        warmup_iters = len(self.train_loader) * LR_WARMUP_EPOCHS
        self.warmup_scheduler = WarmUpLR(self.optimizer, warmup_iters)
        
        # Cosine annealing scheduler for remaining epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - LR_WARMUP_EPOCHS,
            eta_min=0
        )
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, current_lr = self.train_epoch(epoch)
            
            # Update main scheduler after warmup period
            if epoch > LR_WARMUP_EPOCHS:
                self.scheduler.step()
            
            # Validate
            test_loss, test_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)
            
            # Log epoch results
            log_msg = (
                f"Epoch [{epoch:3d}/{num_epochs}] | "
                f"Time: {epoch_time:.1f}s | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}%"
            )
            
            # Check for best accuracy
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_epoch = epoch
                log_msg += " <- Best!"
                
                # Save best model
                if SAVE_BEST_MODEL:
                    self.save_checkpoint(epoch, test_acc, is_best=True)
            
            self.write_log(log_msg)
            
            # Save checkpoint every N epochs
            if epoch % SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(epoch, test_acc, is_best=False)
            
            # Check if target achieved
            if test_acc >= TARGET_ACCURACY and epoch > 50:
                self.write_log(f"\n✓ Target accuracy {TARGET_ACCURACY}% achieved at epoch {epoch}!")
        
        total_time = time.time() - start_time
        
        # Final summary
        self.write_log("\n" + "="*80)
        self.write_log("TRAINING COMPLETED")
        self.write_log("="*80)
        self.write_log(f"Total Training Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        self.write_log(f"Best Test Accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
        self.write_log(f"Final Test Accuracy: {test_acc:.2f}%")
        self.write_log(f"Target Achieved ({TARGET_ACCURACY}%): {'✓ Yes' if self.best_acc >= TARGET_ACCURACY else '✗ No'}")
        self.write_log("="*80)
        
        return self.history
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'best_acc': self.best_acc,
            'history': self.history
        }
        
        if is_best:
            filepath = os.path.join(LOG_DIR, 'best_model.pth')
            torch.save(checkpoint, filepath)
            print(f"✓ Best model saved with accuracy: {accuracy:.2f}%")
        else:
            filepath = os.path.join(LOG_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, filepath)


def main():
    """Main function to run training"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device configuration
    if USE_MPS and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    print("\n" + "="*80)
    print("CIFAR-100 ResNet-18 TRAINING")
    print("="*80 + "\n")
    
    # Print augmentation info
    print_augmentation_info()
    
    # Create model
    print("Creating ResNet-18 model...")
    model = resnet18(num_classes=NUM_CLASSES)
    
    # Print model summary
    total_params = get_model_summary(model, device=device)
    
    # Get data loaders
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        root=DATA_ROOT
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )
    
    # Start training
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Log file saved to: {trainer.log_file}")
    print(f"✓ Best model saved to: {os.path.join(LOG_DIR, 'best_model.pth')}")
    print(f"✓ Best accuracy achieved: {trainer.best_acc:.2f}%")


if __name__ == "__main__":
    main()

