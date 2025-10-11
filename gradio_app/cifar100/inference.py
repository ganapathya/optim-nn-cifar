"""
Inference utilities for ResNet-18 CIFAR-100 model
"""

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import resnet18
from utils import CIFAR100_CLASSES
from config import CIFAR100_MEAN, CIFAR100_STD, NUM_CLASSES


class CIFAR100Predictor:
    """Predictor class for CIFAR-100 ResNet-18 model"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (auto-detected if None)
        """
        # Device configuration
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self.device = device
        
        # Load model
        self.model = resnet18(num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Model accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Test transform (normalization only)
        self.transform = A.Compose([
            A.Resize(32, 32),  # Ensure 32x32
            A.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
            ToTensorV2(),
        ])
        
        print(f"Predictor initialized on {device}")
    
    def preprocess_image(self, image):
        """
        Preprocess a PIL Image or numpy array
        
        Args:
            image: PIL Image or numpy array (H, W, C)
        
        Returns:
            Preprocessed tensor (1, 3, 32, 32)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Apply transform
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image, top_k=5):
        """
        Predict class for a single image
        
        Args:
            image: PIL Image or numpy array
            top_k: Number of top predictions to return
        
        Returns:
            List of tuples (class_name, class_id, probability)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_id = idx.item()
            class_name = CIFAR100_CLASSES[class_id]
            probability = prob.item()
            results.append((class_name, class_id, probability))
        
        return results
    
    def predict_batch(self, images, top_k=5):
        """
        Predict classes for a batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            top_k: Number of top predictions to return per image
        
        Returns:
            List of prediction results (one per image)
        """
        # Preprocess all images
        image_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions for each image
        batch_results = []
        for i in range(len(images)):
            top_probs, top_indices = torch.topk(probabilities[i], top_k)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                class_id = idx.item()
                class_name = CIFAR100_CLASSES[class_id]
                probability = prob.item()
                results.append((class_name, class_id, probability))
            
            batch_results.append(results)
        
        return batch_results


def load_predictor(model_path='logs/cifar100/best_model.pth', device=None):
    """
    Convenience function to load predictor
    
    Args:
        model_path: Path to model checkpoint
        device: Device to use (auto-detected if None)
    
    Returns:
        CIFAR100Predictor instance
    """
    return CIFAR100Predictor(model_path, device)


if __name__ == "__main__":
    import sys
    
    # Test the predictor
    print("Testing CIFAR-100 predictor...")
    
    model_path = 'logs/cifar100/best_model.pth'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    try:
        predictor = load_predictor(model_path)
        print(f"\n✓ Predictor loaded successfully!")
        print(f"✓ Model is ready for inference")
        print(f"\nUsage:")
        print(f"  from cifar100.inference import load_predictor")
        print(f"  predictor = load_predictor('path/to/model.pth')")
        print(f"  results = predictor.predict(image, top_k=5)")
        print(f"  for class_name, class_id, prob in results:")
        print(f"      print(f'{{class_name}}: {{prob:.2%}}')")
        
    except FileNotFoundError:
        print(f"\n✗ Model file not found: {model_path}")
        print(f"  Please train the model first by running: python cifar100/train.py")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")

