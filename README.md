# FixMatch Implementation on CIFAR-10

This repository contains an implementation of the FixMatch algorithm for semi-supervised learning using the CIFAR-10 dataset. FixMatch is a state-of-the-art semi-supervised learning method that combines consistency regularization and pseudo-labeling.

## Overview

FixMatch uses a combination of weak and strong augmentations to generate pseudo-labels for unlabeled data. The algorithm enforces consistency between weakly-augmented and strongly-augmented versions of the same image when the model's predictions are confident.

### Key Features

- Semi-supervised learning approach
- Implementation of FixMatch algorithm on CIFAR-10
- Data augmentation strategies (weak and strong)
- Pseudo-labeling mechanism
- Training visualization and metrics tracking

## Requirements

```bash
tensorflow
numpy
matplotlib
```

## Project Structure

```
fixmatch/
│
├── FixMatch.ipynb        # Main implementation notebook
├── README.md            # This file
├── mymodel.weights.h5   # Saved model weights
```

## Implementation Details

1. **Data Preparation**
   - CIFAR-10 dataset loading and preprocessing
   - Split into labeled and unlabeled sets (80% unlabeled)
   - Data normalization and augmentation

2. **Model Architecture**
   - CNN-based architecture with:
     - 3 Convolutional layers
     - MaxPooling layers
     - Dense layers
     - Final softmax layer

3. **Training Process**
   - Weak augmentation: Random flip, brightness, and contrast
   - Pseudo-label generation with confidence thresholding
   - Strong augmentation: Additional saturation and hue adjustments
   - Combined supervised and unsupervised losses

## Usage

1. Open `FixMatch.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. Monitor training progress through loss and accuracy plots
4. Best model weights will be saved as 'model.weights.h5'

## Results Visualization

The notebook includes visualizations for:
- Training and test accuracy
- Loss components (labeled and unlabeled)
- Number of pseudo-labels generated
- Sample images with augmentations

## Hyperparameters

- Unlabeled batch size ratio (μ): 7
- Confidence threshold (τ): 0.95
- Learning rate: 1e-4
- Training epochs: 500
- Batch size: 32 (labeled)

## References

- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
