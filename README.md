# Neural Network From Scratch

A dataset-independent feed-forward neural network library written from scratch in C++. The project implements the underlying neural network and matrix operations without machine-learning frameworks or external linear algebra libraries.

## Features

- Configurable feed-forward network architecture
- Training & inference
- Custom matrix implementation
- Forward propagation and backpropagation
- Dataset-independent matrix interface
- Designed to support additional activation functions 
- Binary model serialization & deserialization
- No external ML or linear algebra libraries

## Implementation

The library is organized around three abstractions:

Matrix → Layer → Network

Matrix: Provides the numerical operations used throughout the library.

Layer: Manages the parameters and state of an individual network layer.

Network: Composes layers and provides training, prediction, and model management.

External datasets are converted into matrices before being passed to the network, keeping the core implementation independent of any particular dataset.

## MNIST Example

An MNIST example demonstrates training and inference using the library.

The included MNIST `example.nn`, with the following architecture and configuration, achieved **96.86% accuracy** on the 10,000-image MNIST test set.

Architecture: 784 → 128 → 64 → 10  
Training Samples: 60,000  
Epochs: 3  
Input Size: 784  
Output Size: 10  
Learning Rate: 0.01  

The repository includes an MNIST loader that converts the original IDX image and label files into matrices compatible with the network.
The repository does **not** include MNIST training and test sets. To use the MNIST example, download the training and testing data/labels and place them in the respective `data/` and `test/` directories.

The MNIST data/labels can be downloaded from this MNIST dataset mirror: https://github.com/cvdfoundation/mnist

## Building

Compile from the main directory.
```bash
cmake -S . -B build
cmake --build build
```

## Future Improvements

- Mini-batch training
- CUDA acceleration
- Additional activation functions and optimizers
