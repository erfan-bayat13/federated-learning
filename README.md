# Federated Learning for Image and Text Classification

## Project Overview

This project explores the use of federated learning techniques for image and text classification tasks. We utilize two datasets: CIFAR-100 for image recognition and the Shakespeare dataset for text analysis. The primary goal is to compare the performance of a centralized baseline model with a federated learning approach.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Method](#method)
- [Results](#results)
- [References](#references)

## Introduction

Federated learning is a machine learning paradigm where the training process is distributed across multiple devices while keeping the data localized on these devices. This approach enhances data privacy and security, making it particularly useful for sensitive applications.

In this project, we implement and compare a centralized machine learning model and a federated learning model for:
- Image classification using the CIFAR-100 dataset.
- Text classification using the Shakespeare dataset.

## Project Structure
    ```
    federated-learning-project/
    │
    ├── data/
    │   ├── shakespeare/            # shakespeare dataset provided by leaf
    │
    ├── models/                     # model implementations
    │
    ├── experiments/
    │   ├── image_classification/   # Experiments for image classification
    │   └── text_classification/    # Experiments for text classification
    │
    ├── results/                    # Results and performance metrics
    │
    ├── scripts/                    # Utility scripts for data processing and model training
    │
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project readme

    ```

## Datasets

### CIFAR-100
The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images.

### Shakespeare
The Shakespeare dataset is derived from The Complete Works of William Shakespeare. It is used for next-character prediction tasks and consists of text data divided into roles, where each role can be treated as a separate client in federated learning scenarios.


## Installation

To run this project, you need Python 3.7+ and the necessary dependencies listed in `requirements.txt`.

1. Clone the repository:
    ```bash
    git clone https://github.com/erfan-bayat13/federated-learning.git
    cd federated-learning-project
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Ensure that Shakespeare dataset is downloaded and placed in the `data/` directory. You can use the provided scripts in the `scripts/` directory to download and preprocess the datasets.

### Running Experiments

### Experiments

## Image Classification
Centralized: Trains a single model on the entire CIFAR-100 dataset.
Federated: Distributes the CIFAR-100 dataset across multiple simulated clients and trains a federated model.
## Text Classification
Centralized: Trains a single model on the entire Shakespeare dataset.
Federated: Distributes the Shakespeare dataset across multiple simulated clients and trains a federated model.

## Method

In this project, we implemented and compared two federated learning methods: FedAvg and a novel method called FedAvg2Rep.

### FedAvg
FedAvg (Federated Averaging) is a standard federated learning algorithm where each client trains a local model on its own data and sends the model updates to a central server. The server then averages these updates to create a global model, which is sent back to the clients for the next round of training.

### FedAvg2Rep
FedAvg2Rep is an enhancement of the FedAvg algorithm. In this method, after the initial round of federated averaging, a secondary averaging step is introduced to further refine the model. This secondary step aims to improve the representational power of the global model by incorporating additional layers of aggregation, thus potentially leading to better

### Results

The results of the experiments, including accuracy, loss, and other performance metrics, are saved in the results/ directory. Detailed analysis and comparison between the centralized and federated models are documented in this section.

### References

This implementation is based on the following papers:

McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
Hsu et al., "Federated Visual Classification with Real-World Data Distribution", ECCV 2020
Caldas et al., "Leaf: A benchmark for federated settings", Workshop on Federated Learning for Data Privacy and Confidentiality 2019
Reddi et al., "Adaptive Federated Optimization", ICLR 2021
