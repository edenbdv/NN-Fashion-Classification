# NN Fashion Classification

This project focuses on classifying images of clothing into 10 different categories using Neural Networks. The classification is performed using the Fashion-MNIST dataset, which consists of 70,000 grayscale images of clothing items.

## Project Overview

The Neural Network Fashion Classification project aims to build classification models for clothing items and handwritten digits using deep learning techniques. The project is divided into three parts:

### Part 1: Neural Network Implementation using NumPy

In this section, a neural network is implemented from scratch using NumPy to classify the MNIST dataset. The MNIST dataset (Modified National Institute of Standards and Technology database) contains a training set of 60,000 images and a test set of 10,000 images of handwritten digits (0-9). The implementation focuses on the core mechanics of neural networks, including forward propagation, loss calculation, and backpropagation. 

- **Activation Function**: The sigmoid function is used as the activation function in the hidden layer and output layer of the network.
- **Loss Function**: The negative log likelihood is used as the loss function, which helps measure the performance of the classification model. 

The model evaluates accuracy on a validation set.


## Part 2: Neural Network Implementation in PyTorch

In this part of the project, we implement a neural network using PyTorch to classify images from the Fashion-MNIST dataset. This dataset consists of 70,000 grayscale images of clothing items categorized into 10 classes.

### Network Architecture

The architecture of the neural network is as follows:

- **Input Layer**: 784 input units (28x28 pixel images flattened into vectors)
- **Hidden Layer 1**: 128 units with ReLU activation
- **Hidden Layer 2**: 64 units with ReLU activation
- **Output Layer**: 10 units with log-softmax activation, providing class probabilities for each clothing category

### Loss Function

The model uses the **Negative Log Likelihood Loss** as the loss function. This loss function is suitable for multi-class classification tasks, as it measures the performance of the output probabilities against the true labels.

### Optimizer

We utilize **Stochastic Gradient Descent (SGD)** as the optimizer for updating the weights of the neural network. SGD is chosen for its simplicity and effectiveness in optimizing neural network parameters.

### Accuracy Evaluation

The model evaluates accuracy on a validation set. The accuracy is calculated based on the proportion of correctly predicted classes compared to the total number of samples in the validation dataset.


### Part 3: Convolutional Neural Network (CNN) Implementation

In this section, a Convolutional Neural Network (CNN) is implemented using PyTorch to enhance classification accuracy by leveraging spatial hierarchies in images. The CNN model also evaluates accuracy on a validation set and calculates training and validation losses for performance monitoring.

## Installation

To run the project, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- PyTorch
- Matplotlib (for visualization)
- Pandas (for data handling)

You can install the required packages using pip:

```bash
pip install numpy torch matplotlib pandas
