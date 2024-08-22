## Iris Flower Species Classification ##

# Overview

This project demonstrates a machine learning classification model to predict the species of Iris flowers based on their measurements. The Iris dataset, containing measurements for sepal length, sepal width, petal length, and petal width, is used to train and evaluate classification models. The goal is to accurately classify Iris flowers into one of three species: Setosa, Versicolor, or Virginica.

# Project Components

1. Data Exploration and Preprocessing:
   Loading Data: Reads the Iris dataset from a CSV file.
   Exploratory Data Analysis (EDA): Includes visualizations such as pair plots, histograms, and correlation heatmaps.
   Data Preprocessing: Encodes categorical target variables, scales features, and splits data into training and testing sets.

2. Model Training and Evaluation:
   Support Vector Machine (SVM): Trains an SVM classifier with linear kernel and evaluates its performance.
   Random Forest Classifier: Trains and evaluates a Random Forest model.
   Hyperparameter Tuning: Uses GridSearchCV to find the best parameters for the SVM model.

3. Feature Engineering:
   Principal Component Analysis (PCA): Applies PCA for dimensionality reduction and visualization.

4. Model Deployment:
   Pickle Serialization: Saves the trained models, label encoder, and scaler using Pickle for future use.
   Streamlit Application: Provides a simple web interface for users to input flower measurements and get predictions using the   trained model.

# Files

1. "iris_classifier_app.py": Streamlit application for interactive classification of Iris flowers.
2. "svm_iris_model.pkl": Pickled SVM model for prediction.
3. "label_encoder.pkl": Pickled LabelEncoder used for encoding species names.
4. "scaler.pkl": Pickled StandardScaler used for feature scaling.

# Requirements

Python 3.x
Streamlit
scikit-learn
pandas
seaborn
matplotlib

