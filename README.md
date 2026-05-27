# HOUSE PRICE PREDICTION USING LINEAR REGRESSION

## 1. PROJECT TITLE
House Price Prediction using Machine Learning (Linear Regression)

## 2. PROBLEM STATEMENT
The objective of this project is to develop a Machine Learning model that predicts house prices based on key features such as square footage, number of bedrooms, and number of bathrooms.

## 3. OBJECTIVE
- To build a regression model for predicting continuous target values (house prices)
- To analyze the relationship between house features and price
- To evaluate model performance using standard metrics

## 4. DATASET DESCRIPTION
A synthetic dataset is created within the project containing the following features:
- Square Footage (sqft)
- Number of Bedrooms
- Number of Bathrooms
- Target Variable: House Price

## 5. METHODOLOGY
The following steps are implemented:
1. Data creation and preprocessing
2. Feature selection
3. Train-test split
4. Model training using Linear Regression
5. Model evaluation using:
   - Mean Squared Error (MSE)
   - R2 Score
6. Prediction on unseen data
7. Model serialization using Pickle

## 6. TECHNOLOGIES USED
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Pickle

## 7. MODEL USED
Linear Regression (Supervised Learning - Regression)

## 8. EVALUATION METRICS
- Mean Squared Error (MSE)
- R-squared (R2 Score)

## 9. RESULTS
The trained model successfully predicts house prices based on input features with reasonable accuracy on test data.

## 10. HOW TO RUN THE PROJECT
Install dependencies:
```bash
pip install -r requirements.txt
python model.py