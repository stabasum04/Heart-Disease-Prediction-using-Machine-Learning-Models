# Heart-Disease-Prediction-using-Machine-Learning-Models
his repository contains a comprehensive analysis and implementation of machine learning models to predict heart disease based on a dataset containing various health-related attributes. The project explores different preprocessing techniques, applies various classification models.

Project Overview
Heart disease is one of the leading causes of death worldwide, making early prediction crucial for effective treatment and management. This project leverages several machine learning algorithms to classify patients as having heart disease or not based on clinical features.

Key Steps and Features
Data Preprocessing:

Outlier Removal: Removed outliers using the Z-score method to ensure data quality.
Encoding Categorical Variables: Converted text columns to numerical values using Label Encoding for binary categories and One-Hot Encoding for multiclass categories.
Feature Scaling: Applied StandardScaler to standardize the feature values for optimal model performance.
Machine Learning Models Implemented:

Support Vector Classifier (SVC): Used to find the optimal hyperplane that separates the data into different classes with maximum margin.
Logistic Regression: A linear model used for binary classification problems.
Random Forest Classifier: An ensemble learning method that builds multiple decision trees and merges them to improve the accuracy and prevent overfitting.
Dimensionality Reduction with PCA:

Applied Principal Component Analysis (PCA) to reduce the dimensionality of the feature space and retrained the models to observe the impact on accuracy and computational efficiency.
Model Evaluation:

Evaluated model performance using metrics such as accuracy to determine the best-performing model before and after applying PCA.
Results
The Random Forest Classifier provided the highest accuracy before applying PCA, while all models experienced a slight drop in accuracy after applying PCA, demonstrating the trade-off between model performance and computational efficiency.
