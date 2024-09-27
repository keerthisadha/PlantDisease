# Plant Leaf Disease Classification Using Transfer Learning with EfficientNetb5

## Overview

This research project focuses on the classification of plant leaf diseases using deep learning techniques. The project utilizes **EfficientNetb5**, a pre-trained convolutional neural network (CNN), along with transfer learning to classify plant leaf diseases effectively. We achieved an impressive **99.20% test accuracy** using **Keras Deep Learning** modules.


## Abstract

The project addresses the growing challenges in agriculture due to the rise of plant diseases. It proposes a deep learning-based solution for classifying diseases in plant leaves using **Convolutional Neural Networks** (CNNs) and **EfficientNetb5** via transfer learning. The solution automates the process of monitoring and identifying diseases, assisting farmers and agricultural professionals in managing crops more effectively.

## Dataset

- **Source**: [Kaggle](https://www.kaggle.com/)
- **Total Images**: 80,000 images of diseased leaves
- **Number of Classes**: 38 classes of plant diseases

## Methodology

The project employs a CNN-based model architecture, EfficientNetb5, and uses transfer learning to leverage pre-trained weights. This approach improves efficiency and eliminates the need to train the model from scratch. The key steps include:

1. **Data Collection**: Collection of plant leaf disease images from Kaggle.
2. **Data Preprocessing**: Standardizing image sizes, normalization, and noise reduction.
3. **Model Architecture Selection**: Using EfficientNetb5 for efficient feature extraction.
4. **Transfer Learning**: Reusing weights from pre-trained models to reduce training time.
5. **Training and Validation**: Using 70,000 training images, 8,300 validation images, and 1,700 test images.
6. **Tuning Hyperparameters**: Optimizing learning rate, batch size, and epochs.
7. **Evaluation Metrics**: Using accuracy, sensitivity, specificity, and loss metrics.
8. **Comparison and Analysis**: Comparison of the model’s performance with other architectures.

## Results

- **Test Accuracy**: 99.20%
- **EfficientNetb5 Performance**: Efficient feature extraction due to its 300+ layers.
- **Training and Validation Loss**: Minimal loss during training with high accuracy.

## Conclusion

This research demonstrates the effectiveness of transfer learning using EfficientNetb5 for plant leaf disease classification. The model’s high accuracy can significantly assist farmers and agricultural professionals in diagnosing and monitoring crop health. Further validation with external datasets ensures the robustness of the model in real-world applications.

## How to Use

1. **Prerequisites**: Ensure you have Python installed with the following libraries:
   - TensorFlow/Keras
   - NumPy
   - OpenCV for image preprocessing

2. **Training the Model**:
   - Download the dataset from Kaggle.
   - Preprocess the images (resize to 224x224 and normalize).
   - Load the EfficientNetb5 model with pre-trained weights.
   - Train the model using the dataset.

3. **Running the Model**:
   - Pass images of plant leaves through the model.
   - The model will predict the disease class.

## Paper Link

For more details, you can refer to the published research paper: [Plant Leaf Disease Classification using Transfer Learning](https://ieeexplore.ieee.org/document/10421367)
