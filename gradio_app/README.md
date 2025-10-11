# CIFAR-100 Image Classifier with ResNet-18

An interactive image classification application using ResNet-18 trained on CIFAR-100 dataset.

## 🎯 Model Performance

- **Architecture:** ResNet-18 (adapted for CIFAR-100)
- **Accuracy:** 73%+ on CIFAR-100 test set
- **Parameters:** ~11 million
- **Training:** 100 epochs with advanced augmentation and regularization

## 🖼️ Supported Classes

The model can classify images into 100 different categories from CIFAR-100:

**Animals:** beaver, dolphin, otter, seal, whale, aquarium fish, flatfish, ray, shark, trout, leopard, lion, tiger, wolf, bear, and more

**Vehicles:** bicycle, bus, motorcycle, pickup truck, train, lawn mower, rocket, streetcar, tank, tractor

**Nature:** cloud, forest, mountain, plain, sea, maple tree, oak tree, palm tree, pine tree, willow tree

**Household:** bed, chair, couch, table, wardrobe, bottle, bowl, can, cup, plate, clock, keyboard, lamp, telephone, television

**Food & Plants:** apple, mushroom, orange, pear, sweet pepper, orchid, poppy, rose, sunflower, tulip

**And many more!**

## 🚀 How to Use

1. Upload an image using the interface
2. The model will automatically classify it
3. View the top 5 predictions with confidence scores

**Note:** The model works best with small images. Large images will be automatically resized to 32×32 pixels (CIFAR-100's native resolution).

## 🏗️ Model Architecture

ResNet-18 adapted for CIFAR-100:

- Modified initial convolution layer (3×3 instead of 7×7)
- Removed initial max pooling (not needed for 32×32 images)
- Four residual blocks: [2, 2, 2, 2] layers
- Global average pooling + fully connected layer for 100 classes

## 📊 Training Details

**Optimizer:** SGD with Nesterov momentum

- Learning rate: 0.1 (cosine annealing)
- Momentum: 0.9
- Weight decay: 5e-4

**Data Augmentation:**

- Random crop with padding
- Random horizontal flip
- Cutout (random 16×16 patch removal)

**Regularization:**

- Label smoothing (0.1)
- Weight decay
- Batch normalization

**Training Time:** ~2-3 hours on Apple Silicon (M1/M2)

## 📁 Repository

Full code, training scripts, and documentation: [GitHub Repository](https://github.com/yourusername/optim-nn-sifar)

## 🎓 About CIFAR-100

CIFAR-100 is a widely-used benchmark dataset in computer vision consisting of:

- 50,000 training images
- 10,000 test images
- 100 fine-grained classes
- 20 superclasses
- 32×32 pixel resolution

## 📝 Citation

If you use this model, please cite:

```
@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  year={2009}
}
```

## 📄 License

This project is open source and available for educational purposes.

---

Built with ❤️ using PyTorch and Gradio
