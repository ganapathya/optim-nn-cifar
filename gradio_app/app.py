"""
Gradio application for CIFAR-100 ResNet-18 classifier
Deployed on Huggingface Spaces
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path to import cifar100 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cifar100.model import resnet18
from cifar100.utils import CIFAR100_CLASSES
from cifar100.inference import CIFAR100Predictor


# Model path (adjust based on deployment)
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model.pth')


# Initialize predictor globally
print("Loading model...")
try:
    predictor = CIFAR100Predictor(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None


def predict_image(image):
    """
    Predict CIFAR-100 class for uploaded image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Dictionary with class names as keys and probabilities as values
    """
    if predictor is None:
        return {"Error": "Model not loaded"}
    
    if image is None:
        return {"Error": "No image provided"}
    
    try:
        # Get top-5 predictions
        results = predictor.predict(image, top_k=5)
        
        # Format results for Gradio
        predictions = {}
        for class_name, class_id, probability in results:
            # Format: "class_name (ID: class_id)"
            label = f"{class_name.replace('_', ' ').title()}"
            predictions[label] = float(probability)
        
        return predictions
    
    except Exception as e:
        return {"Error": str(e)}


# Create Gradio interface
def create_interface():
    """Create and configure Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background-color: #2563eb !important;
    }
    footer {
        display: none !important;
    }
    """
    
    # Title and description
    title = "🖼️ CIFAR-100 Image Classifier"
    description = """
    ## ResNet-18 trained on CIFAR-100 Dataset
    
    Upload an image and the model will predict which of the 100 CIFAR-100 classes it belongs to.
    The model shows the top 5 predictions with confidence scores.
    
    **Model Details:**
    - Architecture: ResNet-18 (adapted for CIFAR-100)
    - Dataset: CIFAR-100 (60,000 32x32 color images in 100 classes)
    - Training: 100 epochs with SGD, cosine annealing, and label smoothing
    - Accuracy: 73%+ on test set
    
    **Note:** This model works best with small images (32x32). Large images will be resized.
    """
    
    article = """
    ### About CIFAR-100
    
    CIFAR-100 is a challenging image classification dataset consisting of 60,000 32×32 color images in 100 classes, 
    with 600 images per class. The 100 classes are grouped into 20 superclasses.
    
    **Sample Classes:**
    - Animals: beaver, dolphin, otter, seal, whale, etc.
    - Vehicles: bicycle, bus, motorcycle, pickup truck, train, etc.
    - Nature: cloud, forest, mountain, plain, sea, etc.
    - Food: apple, mushroom, orange, pear, sweet pepper, etc.
    - And many more!
    
    ### Model Architecture
    
    This ResNet-18 model has been specifically adapted for CIFAR-100's 32×32 images:
    - Modified initial convolution (3×3 instead of 7×7)
    - No initial max pooling layer
    - ~11 million parameters
    - Trained from scratch (no pre-training)
    
    ### Training Strategy
    
    - **Optimizer:** SGD with Nesterov momentum (0.9)
    - **Learning Rate:** 0.1 with cosine annealing
    - **Augmentations:** Random crop, horizontal flip, cutout
    - **Regularization:** Weight decay (5e-4), label smoothing (0.1)
    - **Training Time:** ~2-3 hours on Apple Silicon
    
    ---
    
    **Repository:** [GitHub](https://github.com/yourusername/optim-nn-sifar)
    """
    
    # Create example images (if available)
    examples = None  # Can add example images here
    
    # Create interface
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
        title=title,
        description=description,
        article=article,
        examples=examples,
        css=custom_css,
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface


# Launch the app
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

