# Deep Learning Examples with TensorFlow and Keras

This repository contains deep learning examples using **TensorFlow** and **Keras**. The examples are organized into separate packages for **Neural Networks** and **Convolutional Neural Networks (CNN)**. Each package includes model files and Colab notebooks for training and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Neural Network Examples](#neural-network-examples)
  - [Basic Neural Network (Sequential API)](#basic-neural-network-sequential-api)
  - [Neural Network using Keras Functional API](#neural-network-using-keras-functional-api)
  - [Multiple Output Model](#multiple-output-model)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [CNN on Horse vs Human Dataset](#cnn-on-horse-vs-human-dataset)
- [Evaluation and Predictions](#evaluation-and-predictions)
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

### Multiple Output Model

This example demonstrates how to define a neural network with multiple outputs, where one output is for classification and another is for regression.

- The model is designed with two separate output layers: one for multi-class classification (using softmax) and another for a continuous value regression (using linear activation).
- Both outputs are trained simultaneously with different loss functions and metrics.
- This is useful for problems where multiple types of predictions need to be made from the same model.

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



## Project Structure

The project is organized as follows:

```
deep-learning-examples/
├── neural_network/                        # Neural Network models (Feedforward and Multi-output models)
│   ├── data/                              # Dataset files for neural network
│   ├── models/                             # Neural network model files
│   │   ├── basic_nn.py                    # Basic Neural Network using Keras Sequential API
│   │   ├── functional_nn.py               # Neural Network using Keras Functional API
│   │   └── multiple_output_nn.py          # Model with multiple outputs
│   └── notebooks/                         # Colab notebooks for neural networks
│       ├── nn_sequential.ipynb            # Colab notebook for Sequential Model
│       ├── nn_functional.ipynb            # Colab notebook for Functional Model
│       └── multiple_output_model.ipynb    # Colab notebook for Multiple Output Model
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


