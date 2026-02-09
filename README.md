# Deep Learning Mastery: From Student to Architect

A comprehensive deep learning curriculum covering everything from fundamental theory to production-grade MLOps. Built with modern TensorFlow 2.x / Keras.

## Who Is This For?

| Level | You Should Know | You Will Learn |
|-------|----------------|----------------|
| **Beginner** | Basic Python, high-school math | Neural network fundamentals, CNNs, training basics |
| **Intermediate** | ML basics, linear algebra | Advanced architectures, transfer learning, regularization |
| **Advanced** | Deep learning concepts, model building | Custom architectures, optimization, Grad-CAM, attention |
| **Architect** | DL development, system design | MLOps, production deployment, scaling, monitoring |

## Learning Path

```
START HERE
    |
    v
[00] TensorFlow_Sample.ipynb
     Modern TensorFlow 2.x Fundamentals
     - Tensors, GradientTape, Eager Execution
     - Sequential, Functional, Subclassing APIs
     - Custom training loops, callbacks, tf.data
     |
     v
[01] 01_Deep_Learning_Theory_Fundamentals.ipynb
     Mathematical & Theoretical Foundations
     - Neurons, forward/backpropagation from scratch
     - Loss functions, optimizers (SGD → Adam)
     - Regularization, batch normalization
     - Convolution operations, attention mechanism
     |
     v
[02] DeepLearningProject_Assignment.ipynb
     Hands-On CNN Project: Pet Classifier
     - Data pipelines & augmentation
     - Basic CNN → Enhanced CNN → Transfer Learning
     - Model evaluation, Grad-CAM visualization
     - Hyperparameter tuning
     - Production export (SavedModel, TF Lite)
     |
     v
[03] 02_Advanced_Architectures_Deep_Dive.ipynb
     Architecture Evolution & Implementation
     - VGG, ResNet, Inception, DenseNet
     - EfficientNet, ConvNeXt
     - Vision Transformers (ViT), Swin Transformer
     - Build each from scratch + compare pretrained
     |
     v
[04] 03_MLOps_Production_Deployment.ipynb
     Production ML Engineering
     - Experiment tracking & reproducibility
     - Model versioning & registry
     - TF Lite optimization & quantization
     - TF Serving, REST/gRPC APIs
     - Monitoring, drift detection, CI/CD for ML
     - Distributed training & scaling
```

## Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge at the top of any notebook to run it directly in your browser with free GPU access.

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/SamuelVinay91/Deeplearning.git
cd Deeplearning

# Create virtual environment
python -m venv dl_env
source dl_env/bin/activate  # Linux/Mac
# dl_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Basic Python**: variables, functions, classes, list comprehensions
- **NumPy basics**: arrays, operations, broadcasting
- **Math**: linear algebra basics (matrices, dot products), calculus (derivatives)

## Notebook Descriptions

### `TensorFlow_Sample.ipynb` - TensorFlow 2.x Fundamentals
Your starting point. Covers the modern TensorFlow ecosystem with eager execution, automatic differentiation via `GradientTape`, three ways to build models (Sequential, Functional, Subclassing), custom training loops, the `tf.data` pipeline API, and practical examples including MNIST classification and linear regression.

### `01_Deep_Learning_Theory_Fundamentals.ipynb` - Theory & Math
The theoretical backbone. Implements a neural network from scratch using only NumPy to build deep intuition. Covers activation functions, forward/backpropagation, loss functions, optimization algorithms (with visualizations of optimizer trajectories), regularization techniques, convolution operations, weight initialization, and an introduction to the attention mechanism.

### `DeepLearningProject_Assignment.ipynb` - CNN Pet Classifier Project
The core hands-on project. Progressively builds three models of increasing sophistication:
1. **Basic CNN** (Beginner) - Sequential model with Conv2D layers
2. **Enhanced CNN** (Intermediate) - BatchNorm, learning rate scheduling, callbacks
3. **Transfer Learning** (Advanced) - MobileNetV2 fine-tuning

Includes data augmentation pipelines, comprehensive model evaluation with confusion matrices, Grad-CAM visualization for model interpretability, and production export guides.

### `02_Advanced_Architectures_Deep_Dive.ipynb` - Architecture Mastery
Deep dive into the evolution of CNN architectures. Implements VGG-16, ResNet (with residual blocks), Inception modules, DenseNet blocks, EfficientNet (with Squeeze-and-Excitation), ConvNeXt blocks, and Vision Transformers from scratch. Includes a practical comparison using `tf.keras.applications` pretrained models.

### `03_MLOps_Production_Deployment.ipynb` - Production Engineering
Bridges the gap between notebook experiments and production systems. Covers experiment configuration management, production-grade `tf.data` pipelines, model versioning/registry, TF Lite conversion with quantization benchmarks, TF Serving with REST/gRPC, monitoring with data/prediction drift detection, automated model testing, CI/CD pipelines, and distributed training strategies.

## Key Topics Covered

### Fundamentals
- Tensors, automatic differentiation, computational graphs
- Neural network architecture (forward pass, backpropagation)
- Activation functions (ReLU, GELU, Swish, Sigmoid, Tanh)
- Loss functions (Cross-Entropy, MSE, Focal Loss)

### Computer Vision
- Convolutional Neural Networks (CNNs)
- Data augmentation and preprocessing pipelines
- Transfer learning and fine-tuning
- Feature map and Grad-CAM visualization

### Modern Architectures
- ResNet (skip connections, identity mappings)
- EfficientNet (compound scaling, MBConv, SE blocks)
- ConvNeXt (modernized CNNs with transformer ideas)
- Vision Transformers (patch embedding, self-attention)
- Swin Transformer (shifted window attention)

### Optimization
- Optimizers: SGD, Momentum, Adam, AdamW
- Learning rate scheduling (cosine annealing, warmup)
- Regularization (Dropout, BatchNorm, L1/L2, weight decay)
- Mixed precision training, XLA compilation

### Production & MLOps
- Model serialization (SavedModel, TF Lite, quantization)
- Serving infrastructure (TF Serving, Docker, REST/gRPC)
- Monitoring (data drift, prediction drift, alerting)
- CI/CD for ML (automated testing, promotion gates)
- Distributed training (MirroredStrategy, multi-GPU)

## Architecture Decision Guide

| Scenario | Recommended Model |
|----------|-------------------|
| Learning / Prototyping | Basic CNN or MobileNetV2 |
| Production baseline | ConvNeXt or EfficientNetV2 |
| Maximum accuracy (large data) | Swin Transformer or ViT |
| Low-data fine-tuning | ConvNeXt or EfficientNet |
| Mobile / Edge deployment | MobileNetV3 or EfficientNet-Lite |
| Resource-constrained training | ResNet with modern recipe |

## Technologies Used

- **TensorFlow 2.x** / **Keras** - Primary deep learning framework
- **NumPy** - Numerical computing and from-scratch implementations
- **Matplotlib** - Visualization and plotting
- **OpenCV** - Image processing
- **TensorFlow Lite** - Mobile/edge model optimization
- **TensorFlow Serving** - Production model serving
- **TensorBoard** - Experiment tracking and visualization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-content`)
3. Commit your changes (`git commit -m 'Add new content'`)
4. Push to the branch (`git push origin feature/new-content`)
5. Open a Pull Request

## License

This project is open source and available for educational purposes.

## Acknowledgments

- TensorFlow team for the excellent framework and documentation
- Original research papers for each architecture covered
- The deep learning community for open educational resources
