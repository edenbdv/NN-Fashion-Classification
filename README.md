# NN Fashion Classification

This project focuses on classifying images of clothing into 10 different categories using Neural Networks. The classification is performed using the Fashion-MNIST dataset, which consists of 70,000 grayscale images of clothing items.

## Project Overview

The Neural Network Fashion Classification project aims to build classification models for clothing items and handwritten digits using deep learning techniques. The project is divided into three parts:

## Part 1: Neural Network Implementation using NumPy

In this section, a neural network is implemented from scratch using NumPy to classify the MNIST dataset. The MNIST dataset (Modified National Institute of Standards and Technology database) contains a training set of 60,000 images and a test set of 10,000 images of handwritten digits (0-9). The implementation focuses on the core mechanics of neural networks, including forward propagation, loss calculation, and backpropagation. 

- **Activation Function**: The sigmoid function is used as the activation function in the hidden layer and output layer of the network.
- **Loss Function**: The negative log likelihood is used as the loss function, which helps measure the performance of the classification model. 

The model evaluates accuracy on a validation set.


## Part 2: Neural Network Implementation in PyTorch

In this part of the project, we implement a neural network using PyTorch to classify images from the Fashion-MNIST dataset. This dataset consists of 70,000 grayscale images of clothing items categorized into 10 classes.
<img width="432" alt="bag" src="https://github.com/user-attachments/assets/6ce3543c-2b52-4fa5-926a-3e4b6f34b9df">
<img width="645" alt="‏‏loss" src="https://github.com/user-attachments/assets/34f09f6d-57d8-4aad-b1e2-fb4c72f29e62">

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


## Part 3: Convolutional Neural Network (CNN) Implementation

In this part of the project, we implement two different Convolutional Neural Network (CNN) architectures using PyTorch to classify images from the Fashion-MNIST dataset. Both architectures leverage convolutional layers, ReLU activation, and max pooling to enhance the model's ability to learn spatial hierarchies from images.

### CNN Architecture 1

The first CNN architecture consists of the following layers:

1. **Convolutional Layers**: Convolves the input image with a set of filters.
2. **ReLU Activation**: Applies the ReLU activation function to introduce non-linearity.
3. **Max Pooling Layers**: Reduces the spatial dimensions of the feature maps, maintaining the most significant features.
   
### Optimizer

We utilize **Stochastic Gradient Descent (SGD)** as the optimizer for updating the weights of the neural network. SGD is chosen for its simplicity and effectiveness in optimizing neural network parameters.

### CNN Architecture 2

The second CNN architecture builds upon the first by incorporating additional components to improve performance:

1. **Convolutional Layesr**: Similar to Architecture 1, it convolves the input image with filters.
2. **ReLU Activation**: Non-linearity is introduced using the ReLU function.
3. **Max Pooling Layers**: Reduces the feature map dimensions.
4. **Batch Normalization**: Normalizes the outputs of the previous layer to stabilize and accelerate training.
5. **Dropout**: A regularization technique that randomly sets a fraction of the input units to zero during training to prevent overfitting.

### Optimizer

For the second CNN architecture, we use the **Adam optimizer**, known for its efficiency in training deep learning models. It adjusts the learning rate based on the first and second moments of the gradients, resulting in faster convergence.

### Summary

In this part, we implement two CNN architectures for classifying clothing images from the Fashion-MNIST dataset. The first architecture establishes a foundational model, while the second architecture enhances performance through the addition of batch normalization, dropout, and the Adam optimizer. These improvements aim to increase the model's robustness and accuracy on the validation set.

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
