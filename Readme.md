# Emotion Analysis in Text

## Overview
This project develops a machine learning model for sentiment analysis, focusing on identifying and classifying emotions in text data. It uses a dataset combining emotions from Hugging Face and Twitter, employing TensorFlow and Keras, with extensive preprocessing and optimization.

## Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, nltk, scikit-learn, TensorFlow, Google Colab

## Installation
1. Clone the repository.
2. Install the required Python packages:
Data Preparation
The dataset contains 31,490 entries with 'Review' and 'Label' columns. The text is preprocessed to remove non-English characters, numbers, special symbols, and stopwords. Labels are simplified and standardized.

##Model Architecture
Embedding Layer: Transforms words into vector representations.
Bidirectional GRU Layers: Learns sequences in both directions.
Dense Layer with ReLU Activation: Applies L2 regularization.
Dropout Layer: Mitigates overfitting.
Output Layer: Multi-class classification with softmax activation.
Adam Optimizer: Custom learning rate.
Training Process
Early stopping on validation accuracy.
Trains for up to 20 epochs with a batch size of 64.
Converges within 10 epochs, indicating efficient tuning.
Evaluation and Results
Final validation accuracy: ~89.85%.
Concludes with a validation loss of 0.3628.
Training and validation accuracy and loss plots provided.
Hyperparameter Tuning
Multiple experiments with different learning rates, epochs, and architectures, highlighting the importance of careful hyperparameter tuning.

##Running Predictions on Test Data
Includes predict_sentiment function for new text data predictions. Demonstrates practical application through example feedback predictions.

##Proof-of-Concept Dashboard
Implements a POC dashboard for visualizing model outputs, providing an interface for sentiment analysis of student feedback.

##Conclusion
The project underlines the importance of learning rates, epochs, and architecture choices in sentiment analysis, with high accuracy in emotion classification.

##Running Predictions on a Test Dataset
Processes a test dataset and saves predictions in emotions.json.
Example predictions showcase feedback sentiment analysis capabilities.
Future Work
Expand the dataset with more diverse text sources.
Explore deeper neural network architectures.
Integrate with real-time data sources for dynamic analysis.
Authors
