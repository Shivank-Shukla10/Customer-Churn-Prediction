# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

The project covers the complete ML lifecycle:

- Data Cleaning
- Exploratory Data Analysis
- Hypothesis Testing
- Feature Engineering
- Classification
- Logistic Regression
- XGBoost
- Hyperparameter Tuning
- Classification Threshold Optimization
- SHAP Explainability
- FastAPI
- Docker

---

## Problem Statement

Customer churn is a major challenge for telecom companies.

The objective of this project is to predict customers who are likely to leave the service so that businesses can take proactive retention actions.

The model predicts:

- `0` → Customer will not churn
- `1` → Customer is likely to churn

---

## Dataset

Telco Customer Churn dataset.

The dataset contains customer demographic information, services, contract details, billing information and churn status.

### Dataset Information

- Records: 7,043
- Features: 20
- Target: Churn

---

## Exploratory Data Analysis

Major findings:

- Month-to-month customers have significantly higher churn.
- Customers with longer tenure are less likely to churn.
- Higher monthly charges are associated with increased churn risk.
- Contract type is one of the strongest churn-related factors.

### Contract vs Churn

| Contract       | Churn Rate |
| -------------- | ---------- |
| Month-to-month | 42.71%     |
| One year       | 11.27%     |
| Two year       | 2.83%      |

Churned customers had an average tenure of approximately 18 months, compared with approximately 38 months for customers who did not churn.

---

## Hypothesis Testing

### Hypothesis 1 — Contract Type vs Churn

Chi-square test was used.

- H0: Contract type and churn are independent.
- H1: Contract type and churn are associated.

The analysis showed a statistically significant relationship between contract type and churn.

### Hypothesis 2 — Tenure vs Churn

Welch's independent two-sample t-test was used.

- H0: Mean tenure is the same for churned and non-churned customers.
- H1: Mean tenure differs between the two groups.

The analysis showed a statistically significant difference in tenure between the two groups.

---

## Machine Learning Models

Two primary classification approaches were evaluated:

1. Logistic Regression
2. XGBoost

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------ | ------- |
| Logistic Regression | 80.70%   | 66.04%    | 56.15% | 60.69% | 84.22%  |
| XGBoost             | 79.56%   | 64.33%    | 51.60% | 57.27% | 83.80%  |
| Tuned XGBoost       | 75.51%   | 52.58%    | 79.14% | 63.18% | 84.22%  |

Class imbalance was handled using `scale_pos_weight`.

---

## Threshold Optimization

Instead of using the default classification threshold of 0.5, the precision-recall curve was used to identify an F1-optimal threshold.

### Final Threshold

```text
0.52529
```
