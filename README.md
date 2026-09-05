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

**Matrix → Layer → Network**

**Matrix:** Provides the numerical operations used throughout the library, including matrix multiplication, transposition, and scalar operations.

**Layer:** Manages the weights, biases, activation functions, and intermediate values required for forward propagation and backpropagation.

**Network:** Composes layers and provides training, inference, and model management.

External datasets are converted into matrices before being passed to the network, keeping the core implementation independent of any particular dataset.

## Training

Training is implemented using forward propagation, backpropagation, and gradient descent.

During forward propagation, each layer computes its weighted inputs and applies its activation function. Intermediate values required for training are cached by each layer and used during backpropagation to calculate gradients for the weights and biases.

Hidden layers currently use ReLU activation, while the output layer uses Softmax for classification. 

## Usage

Create a network by specifying its learning rate and input size, then add layers sequentially to define its architecture.

```cpp
Network neural_net(0.01f, 784);

neural_net.addLayer(128);
neural_net.addLayer(64);
neural_net.addLayer(10, Activation::softmax);
```

Layers default to ReLU activation when no activation function is specified. Training data and labels are provided as vectors of `Matrix` objects.

Train a network by providing training samples, corresponding labels, the number of epochs, and debug level.

```cpp
neural_net.train(training_samples, training_labels, 3, DebugLevel::Standard);
```

Trained networks can be saved to a binary file and loaded later for inference:

```cpp
neural_net.saveNetwork("model.nn");

Network loaded_net = Network::loadNetwork("model.nn");
Matrix output = loaded_net.inference(input);
```

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
