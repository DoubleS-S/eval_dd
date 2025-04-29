# Generated Data Evaluation Framework

A framework for evaluating the performance of generated data compared to distilled data at controlled scales.

## Overview

This project provides tools to evaluate the effectiveness of generated/synthetic data for training machine learning models. The framework specifically allows you to benchmark generated data against distilled data at the same scale (images per class), demonstrating the relative advantages of each approach.

## Features

- Load and evaluate synthetic data from `.npz` files
- Control the number of images per class (IPC) for fair comparison
- Support for various datasets (CIFAR10, MNIST, FashionMNIST, etc.)
- Multiple model architectures (ConvNet, ResNet18, VGG11, etc.) 
- Data augmentation options with Differentiable Siamese Augmentation (DSA)
- Comprehensive evaluation metrics

## Requirements

- PyTorch
- NumPy
- SciPy

## Usage

### Basic Usage

```bash
python eval.py
```

This runs the evaluation with default parameters (CIFAR10, ConvNet, 50 images per class, etc.)

### Custom Evaluation

```bash
python eval.py --generated_data path/to/gen.npz --dataset CIFAR10 --model ConvNet --ipc 50 --num_eval 5
```

### Key Parameters

- `--generated_data`: Path to the generated data file (default: `gen.npz`)
- `--dataset`: Dataset to use (default: `CIFAR10`)
- `--model`: Model architecture (default: `ConvNet`)
- `--ipc`: Images per class (default: `50`)
- `--num_eval`: Number of evaluation runs (default: `3`)
- `--dsa`: Whether to use differentiable Siamese augmentation (default: `True`)

## Data Format

The `gen.npz` file should contain:
- `x`: Array of synthetic images
- `y`: Array of corresponding labels
