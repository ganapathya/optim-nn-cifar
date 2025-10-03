import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import time
import os
from datetime import datetime

from model import CIFAR10Net, get_model_summary
from utils import get_dataloaders, print_augmentation_info


class Trainer:
    """Training manager for CIFAR-10 CNN optimized for Apple Silicon MPS"""
    
    def __init__(self, model, train_loader, test_loader, device, learning_rate=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - using SGD with momentum for better convergence
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        
        # Learning rate scheduler - OneCycleLR for better performance
        self.scheduler = None  # Will be initialized in train()
        
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
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Log file path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'training_log_{timestamp}.txt')
        
        # Write header to log file
        self.write_log("="*80)
        self.write_log("CIFAR-10 TRAINING LOG")
        self.write_log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_log(f"Device: {device}")
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
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
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
            for inputs, targets in self.test_loader:
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
    
    def train(self, num_epochs=50):
        """Main training loop"""
        self.write_log(f"\nStarting training for {num_epochs} epochs...")
        self.write_log("="*80 + "\n")
        
        # Initialize OneCycleLR scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.1,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.2,
            anneal_strategy='cos'
        )
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, current_lr = self.train_epoch(epoch)
            
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
                self.save_checkpoint(epoch, test_acc, is_best=True)
            
            self.write_log(log_msg)
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, test_acc, is_best=False)
        
        total_time = time.time() - start_time
        
        # Final summary
        self.write_log("\n" + "="*80)
        self.write_log("TRAINING COMPLETED")
        self.write_log("="*80)
        self.write_log(f"Total Training Time: {total_time/60:.2f} minutes")
        self.write_log(f"Best Test Accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
        self.write_log(f"Final Test Accuracy: {test_acc:.2f}%")
        self.write_log(f"Target Achieved (85%): {'✓ Yes' if self.best_acc >= 85.0 else '✗ No'}")
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
            filepath = os.path.join(self.log_dir, 'best_model.pth')
            torch.save(checkpoint, filepath)
            print(f"Best model saved with accuracy: {accuracy:.2f}%")
        else:
            filepath = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, filepath)


def main():
    """Main function to run training"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration - optimized for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("✗ MPS not available, using CPU")
    
    print("\n" + "="*80)
    print("CIFAR-10 CNN TRAINING - APPLE SILICON OPTIMIZED")
    print("="*80 + "\n")
    
    # Print albumentations info
    print_augmentation_info()
    
    # Create model
    print("Creating model...")
    model = CIFAR10Net(num_classes=10)
    
    # Print model summary
    total_params = get_model_summary(model, device=device)
    
    # Get data loaders
    print("Loading CIFAR-10 dataset with albumentations...")
    # For MPS, num_workers=0 or 2 is more stable than 4
    num_workers = 0 if device.type == 'mps' else 4
    train_loader, test_loader = get_dataloaders(
        batch_size=128,
        num_workers=num_workers,  # Optimized for Apple Silicon
        root='./data'
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: 128\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.01
    )
    
    # Start training
    history = trainer.train(num_epochs=50)
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Log file saved to: {trainer.log_file}")
    print(f"✓ Best model saved to: {os.path.join(trainer.log_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()

