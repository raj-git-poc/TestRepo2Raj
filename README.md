# Machine Learning Classification Models – Assignment 2

## 1. Problem Statement

The objective of this assignment is to implement multiple machine learning classification models on a public dataset and compare their performance using standard evaluation metrics. 

Additionally, an interactive Streamlit web application has been developed to:
- Compare model performances
- Allow model selection
- Display evaluation metrics
- Show confusion matrix and classification report
- Allow dataset upload for prediction

This project demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment.

---

## 2. Dataset Description

Dataset Name: Breast Cancer Wisconsin (Diagnostic) Dataset  
Source: UCI Machine Learning Repository  
URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  

The dataset is also available through scikit-learn using:
`sklearn.datasets.load_breast_cancer()`

### Dataset Details:
- Total Instances: 569
- Total Features: 30 numerical features
- Target Classes:
  - 0 → Malignant
  - 1 → Benign
- Type: Binary Classification Problem

The dataset satisfies the assignment requirement:
- Minimum 500 instances ✅
- Minimum 12 features ✅

Before training:
- Data was split into training (80%) and testing (20%)
- Feature scaling was applied using StandardScaler (for models requiring scaling)

---

## 3. Models Used and Evaluation Metrics

The following 6 machine learning models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. Gradient Boosting (Boosting Ensemble – XGBoost equivalent)

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
|-----------|----------|-----|-----------|--------|----------|------------|
| Logistic Regression | (fill from output) | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |
| Gradient Boosting | | | | | | |

(Note: Values are generated automatically in the Streamlit app and saved in model_results.csv)

---

## 5. Observations on Model Performance

| ML Model | Observation |
|-----------|------------|
| Logistic Regression | Performs very well due to linear separability of features. Good balance between bias and variance. |
| Decision Tree | Performs reasonably well but may slightly overfit the training data. |
| KNN | Sensitive to feature scaling. Performs well after normalization. |
| Naive Bayes | Assumes feature independence; performs decently but slightly lower than ensemble models. |
| Random Forest | Strong performance due to ensemble learning and reduction of overfitting. |
| Gradient Boosting | Shows high performance due to sequential boosting and improved learning of complex patterns. |

Overall, ensemble models (Random Forest and Gradient Boosting) generally perform better compared to individual classifiers due to variance reduction and improved generalization.

---

## 6. Streamlit Application Features

The deployed Streamlit application includes:

- Model performance comparison table
- Model selection dropdown
- Automatic training and evaluation
- Confusion Matrix display
- Classification Report display
- Dataset upload option (CSV with target column)

---

## 7. Project Structure

