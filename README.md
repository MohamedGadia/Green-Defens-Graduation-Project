
# Green Defend CNN

This repository contains the implementation of a Convolutional Neural Network (CNN) designed to classify plant diseases from images, using a dataset of healthy and diseased crop leaves.

---

## Project Overview

The goal of this project is to identify 38 different plant diseases from RGB images of crop leaves. By leveraging deep learning techniques, this model helps in detecting and categorizing diseases, aiding farmers and agricultural experts.

### Dataset
- **Source**: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
- **Statistics**:
  - Training Data: 56,251 images
  - Validation Data: 14,044 images (20% of training data)
  - Test Data: 17,572 images
  - Classes: 38 (including healthy and diseased leaves)

---

## Key Features

### 1. Data Preprocessing
- Scaling images using `ImageDataGenerator`.
- Augmentation applied offline to enhance dataset variety.
- Directory-based data loading for efficient memory usage.

### 2. Exploratory Data Analysis (EDA)
- Insights into dataset composition (e.g., class distribution, number of unique plants).
- Visualization of sample images and disease categories.

### 3. CNN Architecture
- Multiple Conv2D layers with ELU activations and GlorotNormal kernel initialization.
- Pooling, Dropout, and Dense layers to optimize model performance.
- Input shape: `(224, 224, 3)`

### 4. Model Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Callbacks:
  - `ModelCheckpoint`: Saves the best-performing model.
  - `EarlyStopping`: Stops training to prevent overfitting.

### 5. Evaluation
- Test accuracy and loss reported.
- Classification report and confusion matrix for performance metrics.

### 6. Predictions
- Visualization of model predictions on random test samples.

---

## Results
- **Train Accuarcy**: Achieved high accuracy during training, demonstrating the model's ability to learn effectively from the data.
- 
- **Validation Accuracy**:Consistently aligned with training accuracy, indicating that the model generalizes well without overfitting.
- 
- **Test Accuracy**: The model achieved an outstanding **98%** accuracy on the test dataset, demonstrating its robustness in correctly classifying plant diseases across unseen data.
- 
- **Key Metrics**: Classification report includes precision, recall, and F1-score for each class.

---

### Key Libraries:
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Author
**Mohamed Ahmed Gadia**  
Email: [mohamedgadia00@gmail.com]  
LinkedIn: [Mohamed Gadia](www.linkedin.com/in/mohamedgadia)  
