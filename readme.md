# Amazon Product Price Prediction

This project focuses on predicting product prices based on a combination of textual descriptions, product images, and structured features. It leverages advanced deep learning models for text and image encoding, followed by a LightGBM regressor for the final price prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)

## Project Overview

This notebook demonstrates an end-to-end machine learning pipeline for predicting product prices. It combines multimodal data (textual content, image links) with structured features to build a robust prediction model. The pipeline includes data loading, feature extraction using pre-trained BERT and EfficientNet models, feature engineering, model training, and prediction generation.

## Features

- **Text Embedding**: Uses `DistilBERT` to encode product `catalog_content` into dense vector representations.
- **Image Embedding**: Uses `EfficientNet_b0` to extract features from product `image_link`s.
- **Structured Features**: Extracts numerical features like `quantity` from the `catalog_content`.
- **Price Prediction**: Employs a `LightGBM` regressor to predict the `price` based on the combined text, image, and structured embeddings.
- **Persistent Storage**: Saves intermediate embeddings and the trained model to Google Drive for reusability and to avoid re-computation.

## Dataset

The project uses two CSV files:
- `train.csv`: Contains training data including `catalog_content`, `image_link`, and `price`.
- `test.csv`: Contains test data with similar features (excluding `price`) for prediction.

These files are expected to be located in your Google Drive at `/content/drive/MyDrive/FinalAmazML/`.

## Setup

To run this notebook, you need a Google Colab environment with GPU access (recommended).

1.  **Mount Google Drive**: The notebook will prompt you to mount your Google Drive to access the dataset and save artifacts.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Install Dependencies**: Ensure all necessary libraries are installed. The notebook starts with common imports, but specific libraries like `transformers`, `timm`, `lightgbm`, and `joblib` are crucial.
    ```python
    !pip install transformers timm lightgbm
    ```

3.  **Data Placement**: Place `train.csv` and `test.csv` in `/content/drive/MyDrive/FinalAmazML/`.

## Usage

Follow the cells in the notebook sequentially:

1.  **Import Libraries**: Run the first code cell to import all required libraries.
2.  **Mount Google Drive**: Execute the drive mounting cell.
3.  **Load Data**
4.  **Encode Text**: Uses `DistilBERT` to encode text. It will load pre-computed embeddings if available or compute and save them.
5.  **Encode Images**: Uses `EfficientNet` to encode images. Similar to text, it loads pre-computed embeddings or computes and saves them.
6.  **Extract Structured Features**: Extracts the `quantity` feature.
7.  **Train Model**: Trains a `LightGBM` regressor. It will load a pre-trained model and processed data if available or train and save them.
8.  **Predict on Test Set**: Prepares the test data and generates price predictions.
9.  **Save Submission**: Saves the predictions to `test_out.csv` in your Google Drive.

## Results

The final output is a CSV file named `test_out.csv` located in `/content/drive/MyDrive/FinalAmazML/`. This file contains two columns: `sample_id` and the predicted `price` for each item in the test set.

## Dependencies

- `pandas`
- `numpy`
- `torch`
- `joblib`
- `requests`
- `Pillow` (PIL)
- `transformers` (for DistilBERT)
- `timm` (for EfficientNet)
- `torchvision`
- `scikit-learn`
- `lightgbm`
- `tqdm`
