# Skin-Cancer-Classifier

## Table of Contents
- [Introduction](#introduction)
- [Techniques Used](#techniques-used)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

---

## Introduction

This project explores the use of digital image processing and machine learning to classify skin lesions as either malignant or benign. Leveraging advanced image enhancement techniques, feature extraction methods, and machine learning algorithms, the project aims to improve the detection accuracy of skin cancer.

The project is based on the public dataset titled **"Skin Cancer: Malignant vs. Benign"** and combines state-of-the-art methodologies to enhance image quality, extract meaningful features, and classify the data.

---

## Techniques Used

### 1. Image Enhancement
- **Histogram Equalization**: Improves contrast in images.
- **Sharpening**: Enhances the edges and fine details.

### 2. Feature Extraction
- **SIFT (Scale-Invariant Feature Transform)**: Identifies key points in images for matching.
- **LBP (Local Binary Patterns)**: Captures texture details for classification.
- **HOG (Histogram of Oriented Gradients)**: Extracts features based on gradient orientations.

### 3. Modeling
- **Pre-trained CNN (VGG16)**: Used for feature extraction and ensemble learning.
- **Ensemble Techniques**: Combines predictions from multiple classifiers for better accuracy.
- **Classifiers**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machines (SVM)

---

## Dependencies

To run this project, ensure you have the following dependencies installed:

- **Python 3.8 or later**: The base programming language used.
- **Jupyter Notebook**: To run and edit the provided `.ipynb` files interactively.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning model implementation.
- **Matplotlib**: For data visualization and plotting.
- **OpenCV**: For image processing and computer vision tasks.
- **Seaborn**: For enhanced data visualizations (optional but recommended).
- **TensorFlow/Keras**: For using the pre-trained VGG16 model.

  ---

  ## Results

This project showcases the effectiveness of combining image processing techniques with machine learning for medical image classification. Detailed visualizations and metrics are included in each notebook to evaluate model performance.
