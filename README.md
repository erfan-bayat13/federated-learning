# Federated Learning for Image and Text Classification

## Project Overview

This project explores the use of federated learning techniques for image and text classification tasks. We utilize two datasets: CIFAR-100 for image recognition and the Shakespeare dataset for text analysis. The primary goal is to compare the performance of a centralized baseline model with a federated learning approach.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)

## Introduction

Federated learning is a machine learning paradigm where the training process is distributed across multiple devices while keeping the data localized on these devices. This approach enhances data privacy and security, making it particularly useful for sensitive applications.

In this project, we implement and compare a centralized machine learning model and a federated learning model for:
- Image classification using the CIFAR-100 dataset.
- Text classification using the Shakespeare dataset.

## Datasets

### CIFAR-100
The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images.

### Shakespeare
The Shakespeare dataset is derived from The Complete Works of William Shakespeare. It is used for next-character prediction tasks and consists of text data divided into roles, where each role can be treated as a separate client in federated learning scenarios.


## Installation

To run this project, you need Python 3.7+ and the necessary dependencies listed in `requirements.txt`.

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/federated-learning-project.git
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

Ensure that the CIFAR-100 and Shakespeare datasets are downloaded and placed in the `data/` directory. You can use the provided scripts in the `scripts/` directory to download and preprocess the datasets.

### Running Experiments

### Experiments

## Image Classification
Centralized: Trains a single model on the entire CIFAR-100 dataset.
Federated: Distributes the CIFAR-100 dataset across multiple simulated clients and trains a federated model.
## Text Classification
Centralized: Trains a single model on the entire Shakespeare dataset.
Federated: Distributes the Shakespeare dataset across multiple simulated clients and trains a federated model.

### Results

The results of the experiments, including accuracy, loss, and other performance metrics, are saved in the results/ directory. Detailed analysis and comparison between the centralized and federated models are documented in this section.
