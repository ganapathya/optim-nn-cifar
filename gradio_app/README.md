---
title: CIFAR-100 ResNet-18 Classifier
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# CIFAR-100 Image Classifier 🖼️

An interactive image classification application using **ResNet-18** trained from scratch on the CIFAR-100 dataset.

## 🎯 Model Performance

- **Accuracy:** 78.33% on CIFAR-100 test set (exceeds 73% target!)
- **Architecture:** ResNet-18 adapted for 32×32 images
- **Parameters:** ~11 million
- **Training:** 100 epochs (~1.5 hours on Apple Silicon)

## 🚀 Try It Out

1. **Upload an image** - Any image format (JPG, PNG, etc.)
2. **Get predictions** - Instant classification into one of 100 categories
3. **See confidence** - Top 5 predictions with probability scores

## 📊 CIFAR-100 Classes

The model can classify images into **100 diverse categories**:

### Animals

- Mammals: beaver, dolphin, otter, seal, whale, elephant, leopard, lion, tiger, wolf, bear, hamster, mouse, rabbit, shrew, squirrel
- Insects: bee, beetle, butterfly, caterpillar, cockroach
- Aquatic: aquarium fish, flatfish, ray, shark, trout
- Reptiles: crocodile, dinosaur, lizard, snake, turtle

### Vehicles

- Ground: bicycle, bus, motorcycle, pickup truck, streetcar, tank, tractor, train
- Other: lawn mower, rocket

### Nature & Outdoors

- Landscapes: cloud, forest, mountain, plain, road, sea
- Trees: maple tree, oak tree, palm tree, pine tree, willow tree
- Flowers: orchid, poppy, rose, sunflower, tulip

### Household Items

- Furniture: bed, chair, couch, table, wardrobe
- Objects: bottle, bowl, can, cup, plate, clock, keyboard, lamp, telephone, television

### Food

- Produce: apple, mushroom, orange, pear, sweet pepper

### People & Buildings

- People: baby, boy, girl, man, woman
- Structures: bridge, castle, house, skyscraper

### And More!

- Animals: camel, cattle, chimpanzee, crab, kangaroo, porcupine, possum, raccoon, skunk, snail, spider
- Small creatures: caterpillar, worm

## 🏗️ Technical Details

### Model Architecture

**ResNet-18 for CIFAR-100:**

- Modified initial convolution: 3×3 (stride=1) instead of 7×7
- No initial max pooling layer (designed for small 32×32 images)
- 4 residual blocks with [2, 2, 2, 2] layers
- Global average pooling + fully connected layer
- Output: 100 classes

### Training Configuration

**Optimizer:**

- SGD with Nesterov momentum (0.9)
- Weight decay: 5e-4
- Initial learning rate: 0.1

**Learning Rate Schedule:**

- Warmup: 5 epochs (linear)
- Cosine annealing: remaining 95 epochs

**Loss Function:**

- CrossEntropyLoss with label smoothing (0.1)

**Data Augmentation:**

- Random crop (32×32) with padding (4)
- Random horizontal flip (p=0.5)
- Cutout (16×16 patch, p=0.5)
- Normalization (CIFAR-100 mean/std)

**Regularization:**

- Weight decay (L2 regularization)
- Label smoothing
- Batch normalization in all residual blocks

### Training Results

```
Epoch [100/100] | Test Acc: 78.29%
Best Test Accuracy: 78.33% (Epoch 96)
Target Achieved (73.0%): ✓ Yes
Training Time: 87.13 minutes (1.45 hours)
```

**Training Progression:**

- Epoch 20: ~55% → Learning fundamentals
- Epoch 50: ~68% → Rapid improvement
- Epoch 80: ~75% → Fine-tuning
- Epoch 96: **78.33%** → Best performance!

## 💻 How It Works

### Image Processing Pipeline

```
Your Image
    ↓
Resize to 32×32
    ↓
Normalize (CIFAR-100 mean/std)
    ↓
ResNet-18 Forward Pass
    ↓
Softmax Probabilities
    ↓
Top-5 Predictions
```

### Inference Speed

- **~10-20ms** per image
- Instant results in the web interface

## 📚 Dataset Information

**CIFAR-100:**

- 60,000 color images (32×32 pixels)
- 100 fine-grained classes
- 20 superclasses
- 600 images per class
- Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton

## 🎓 Key Achievements

✅ **Exceeded Target:** 78.33% vs 73% goal (+5.33%)  
✅ **From Scratch:** No pre-training, fully trained on CIFAR-100  
✅ **Efficient:** Only 1.5 hours training on Apple Silicon  
✅ **Production Ready:** Clean architecture, well-documented code  
✅ **Interactive:** User-friendly Gradio interface

## 🔧 Usage Tips

**Best Results:**

- Images with clear subjects work best
- The model was trained on 32×32 images (low resolution)
- Works with any image size (auto-resized)
- Try images of animals, vehicles, objects, or nature

**Example Images to Try:**

- Photo of a cat → likely predicts "cat" or similar animal
- Picture of a car → might predict "automobile", "pickup_truck", etc.
- Nature scene → could be "forest", "mountain", "sea", etc.

## 📖 Model Card

**Model Name:** ResNet-18 CIFAR-100 Classifier  
**Model Type:** Image Classification (Convolutional Neural Network)  
**Training Data:** CIFAR-100 training set (50,000 images)  
**Test Data:** CIFAR-100 test set (10,000 images)  
**Framework:** PyTorch 2.0+  
**Hardware:** Apple Silicon (MPS)  
**License:** MIT

## 🔗 Resources

- **Code Repository:** [GitHub](https://github.com/yourusername/optim-nn-sifar)
- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Dataset:** [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

## 👨‍💻 About

This model was trained as part of a deep learning project to achieve high accuracy on CIFAR-100 using ResNet architecture. The implementation includes modern training techniques like cosine annealing, label smoothing, and strong data augmentation.

**Training Environment:**

- Device: Apple Silicon (M1/M2/M3)
- Backend: MPS (Metal Performance Shaders)
- Framework: PyTorch with albumentations for augmentation

## 📝 Citation

```bibtex
@article{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={CVPR},
  year={2016}
}

@techreport{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  year={2009}
}
```

## ⚠️ Limitations

- Model trained on 32×32 low-resolution images
- Performance may vary on high-resolution images
- Best suited for CIFAR-100 style images
- May not generalize well to very different image styles

---

**Built with PyTorch, Gradio, and Albumentations**  
**Deployed on Huggingface Spaces** 🤗
