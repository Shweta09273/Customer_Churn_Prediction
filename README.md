
# Customer Churn Prediction

## Overview

Customer Retention Analytics & Churn Prediction System is a machine learning project that predicts whether a customer is likely to churn or stay.

The application provides:

- Churn Prediction
- Churn Probability Score
- Risk Classification
- Retention Recommendations
- Prediction History Dashboard
- Interactive Streamlit

## Project Overview

Customer churn prediction is a machine learning project that predicts whether a customer is likely to leave a telecom company based on customer demographics, account information, services subscribed, and billing details.

The project performs:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression
- Random Forest Classification
- Model Evaluation
- Feature Importance Analysis

---

## Dataset

Dataset: IBM Telco Customer Churn Dataset

Records: 7043 customers

Target Variable:
- Churn (Yes/No)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Project Structure

Customer-Churn-Prediction/

├── data/
│   └── customer_churn.csv

├── images/
│   ├── churn_distribution.png
│   ├── gender_churn.png
│   ├── contract_churn.png
│   ├── internet_churn.png
│   ├── monthly_charges.png
│   ├── tenure_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png

├── notebook/
│   └── churn_model.py

├── README.md

└── requirements.txt

---

## Exploratory Data Analysis

Generated visualizations:

- Churn Distribution
- Gender vs Churn
- Contract vs Churn
- Internet Service vs Churn
- Monthly Charges Distribution
- Tenure Distribution

---

## Models Used

### Logistic Regression

Accuracy: 81.7%

Recall: 58.2%

ROC-AUC: 74.2%

### Random Forest

Accuracy: 79.5%

Recall: 46.9%

ROC-AUC: 69.1%

---

## Confusion Matrix

The confusion matrix visualizes model performance and classification errors.

---

## Feature Importance

Random Forest feature importance identifies the most influential features affecting customer churn.

---

## Future Improvements

- Hyperparameter Tuning
- XGBoost Implementation
- Cross Validation
- Model Deployment using Flask/Streamlit

---
# Customer Retention Analytics & Churn Prediction System

Developed By: Shweta Singh

## Features
- Customer Churn Prediction
- Churn Probability Score
- Risk Classification
- Retention Recommendations
- Prediction History
- Business Dashboard

## Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-Learn
- Matplotlib

## Run Project

pip install -r requirements.txt

streamlit run app.py

