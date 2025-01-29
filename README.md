# Deep Learning Examples with TensorFlow and Keras

This repository contains deep learning examples using **TensorFlow** and **Keras**. The examples are organized into separate packages for **Neural Networks** and **Convolutional Neural Networks (CNN)**. Each package includes model files and Colab notebooks for training and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Neural Network Examples](#neural-network-examples)
  - [Basic Neural Network (Sequential API)](#basic-neural-network-sequential-api)
  - [Neural Network using Keras Functional API](#neural-network-using-keras-functional-api)
  - [Model Subclassing](#model-subclassing)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [CNN on Horse vs Human Dataset](#cnn-on-horse-vs-human-dataset)
- [Evaluation and Predictions](#evaluation-and-predictions)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Project Structure](#project-structure)


## Introduction

This repository provides a collection of examples demonstrating the capabilities of **Keras** and **TensorFlow** for building deep learning models. The models are organized into two main categories:

1. **Neural Networks** (feedforward, multiple output models, etc.)
2. **Convolutional Neural Networks (CNN)** for image classification tasks.

## Neural Network Examples

### Basic Neural Network (Sequential API)

This example demonstrates how to build a simple **Feedforward Neural Network** using Keras' **Sequential API** on the **Iris Dataset** for classification.

The dataset is split into training and testing sets, and a basic feedforward neural network is built using Keras' **Sequential API**. The model is then trained on the Iris dataset and evaluated for performance. The steps covered in this example include:

- Loading the dataset and preprocessing (normalization and one-hot encoding).
- Defining a simple neural network with fully connected layers.
- Training the model and evaluating its accuracy.

### Neural Network using Keras Functional API

In this example, a more complex neural network is built using the **Keras Functional API**. The Functional API allows you to define more flexible model architectures compared to the Sequential API, such as multi-input or multi-output models.

This approach is useful when building models that require shared layers or non-linear architectures. The model is then trained and evaluated on the same Iris dataset, with a more modular and flexible approach.

### Model Subclassing

In this example, you will learn how to build a custom model using the **Keras Subclassing API**. This method allows you to define a model by subclassing the `tf.keras.Model` class and implementing the `__init__` and `call` methods to define the model’s architecture and forward pass.

Model subclassing is particularly useful when your model needs custom behaviors that cannot be easily achieved with the Sequential or Functional APIs. In this example, you will:

- Define a custom neural network by subclassing `tf.keras.Model`.
- Implement the forward pass (in the `call` method) for a simple fully connected neural network.
- Train and evaluate the model on the Iris dataset.


## Convolutional Neural Network (CNN)

### CNN on Horse vs Human Dataset

This example demonstrates how to build a **Convolutional Neural Network (CNN)** for image classification, using the **Horse vs Human dataset**. CNNs are particularly effective for image-related tasks as they can capture spatial hierarchies in data.

- The dataset is preprocessed (resizing images and normalizing pixel values).
- A CNN architecture is built using Keras with convolutional, pooling, and fully connected layers.
- The model is trained on the image data, and evaluated for accuracy on the test set.

## Evaluation and Predictions

Once the models are trained, you can evaluate their performance on unseen data and make predictions. The evaluation step typically involves testing the model on a held-out dataset, while prediction involves using the trained model to make new inferences.

Steps include:
- Evaluating the model using test data to obtain metrics like accuracy or loss.
- Making predictions for new inputs and interpreting the results, including classification labels and regression outputs.

## Activation Functions
Activation functions are essential for introducing non-linearity into neural networks, enabling them to learn complex patterns. Depending on the type of task, different activation functions are used in the output layer:
- **Sigmoid:** The sigmoid activation function is typically used in binary classification tasks. It maps outputs to a range between 0 and 1, which can be interpreted as the probability of one class. It's commonly used in the output layer when there are only two possible classes.
- **Softmax:** The softmax activation function is used in multi-class classification problems. It converts raw output values (logits) into a probability distribution, where each class has a probability between 0 and 1, and the sum of all probabilities equals 1. It is commonly used in the output layer for classification tasks with more than two classes.
- **Linear:** The linear activation function is used for regression tasks, where the output is a continuous value. It outputs real-valued numbers without any transformation, making it suitable for predicting quantities like price, temperature, etc.
- **ReLu(Rectified Linear Unit):** ReLU is one of the most widely used activation functions in hidden layers of deep networks. It outputs the input directly if it is positive; otherwise, it returns zero.

## Loss Functions
Choosing the right loss function is crucial for training a deep learning model, as it determines how the model's predictions are evaluated during training. The loss function depends on the type of task:
- **Binary Crossentropy:** Used for binary classification tasks (e.g., predicting whether an image is of a horse or human). The binary crossentropy loss function compares the predicted probability with the true binary label.
- **Categorical Crossentropy:** Used for multi-class classification tasks (e.g., classifying an image into one of several categories). The categorical crossentropy loss function compares the predicted probability distribution with the true class label, which is usually one-hot encoded.
- **Mean Squared Error(MSE)** Used for regression tasks, where the goal is to predict continuous values (e.g., predicting the price of a house). MSE measures the average of the squared differences between predicted values and actual values.


## Project Structure

The project is organized as follows:

```
deep-learning-examples/
├── neural_network/                        # Neural Network models (Feedforward and Multi-output models)
│   ├── data/                              # Dataset files for neural network
│   ├── models/                             # Neural network model files
│   │   ├── basic_nn.py                    # Basic Neural Network using Keras Sequential API
│   │   ├── functional_nn.py               # Neural Network using Keras Functional API
│   │   └── model_subclassing.py          # Model with multiple outputs
│   └── notebooks/                         # Colab notebooks for neural networks
│       ├── nn_sequential.ipynb            # Colab notebook for Sequential Model
│       ├── nn_functional.ipynb            # Colab notebook for Functional Model
│       └── model_subclassing.ipynb        # Colab notebook for Model Subclassing
├── cnn/                                   # CNN models for image classification
│   ├── data/                              # Dataset files for CNN (e.g., Horse vs Human dataset)
│   ├── models/                             # CNN model file
│   │   └── cnn_model.py                   # CNN model for Horse vs Human classification
│   └── notebooks/                         # Colab notebooks for CNN models
│       └── cnn_horse_human.ipynb          # Colab notebook for CNN model on Horse vs Human dataset
├── requirements.txt                       # Python dependencies (for local setup)
├── README.md                              # Project overview and instructions
└── .gitignore                             # List of files/folders to be ignored by git


```


