# Customer Retention Intelligence

An end-to-end machine learning project that analyzes customer purchasing behavior, predicts customer churn, and helps businesses identify customers who may stop purchasing.

## Overview

Customer retention is important for sustainable business growth. Losing existing customers can affect revenue and increase the cost of acquiring new customers.

**Customer Retention Intelligence** uses historical retail transaction data to build a machine learning model that predicts customer churn. The project combines data preprocessing, customer-level feature engineering, model evaluation, and an interactive Streamlit dashboard to turn transaction data into actionable insights.

The application allows users to explore customer behavior, identify high-risk customers, make churn predictions, and evaluate model performance.

## Live Demo

- **GitHub Repository:** https://github.com/anvitharv-create/Customer-Retention-Intelligence
- **Live Dashboard:** https://customer-retention-intelligence-69jqdy8zdbwmacrgmtvk7d.streamlit.app

## Dashboard Screenshots

### 1. Analytics Overview

Explore customer retention patterns, churn distribution, and key business metrics.

![Analytics Overview](screenshots/analytics-overview.png)

### 2. High-Risk Customers

Identify customers who may be at risk of churning and prioritize them for retention efforts.

![High-Risk Customers](screenshots/high-risk-customers.png)

### 3. Predict Churn

Use the trained machine learning model to estimate the likelihood of customer churn.

![Predict Churn](screenshots/predict-churn.png)

### 4. Model Performance

Review model evaluation results and understand how effectively the model identifies potential churn.

![Model Performance](screenshots/model-performance.png)

## Problem Statement

Retail businesses generate large amounts of transaction data, but historical purchases alone do not directly identify which customers may stop buying.

This project addresses the problem by transforming transaction-level data into customer-level features and using machine learning to identify potential churn. The resulting insights can help businesses prioritize retention campaigns and make data-driven decisions.

## Key Results

| Metric | Result |
|---|---:|
| Cleaned transaction records | 392,692 |
| Customers in the feature dataset | 3,370 |
| Customers labelled as churned | 43.06% |
| Temporal test ROC-AUC | 0.7426 |
| Temporal test recall | 88.84% |
| Temporal test F1-score | 68.18% |

*The model metrics above are from the temporal evaluation of Logistic Regression at a selected probability threshold of 0.25. Results depend on the dataset, split, and threshold.*

## Key Features

- **Customer Analytics:** Explore customer-level data and retention patterns.
- **Churn Prediction:** Predict whether a customer is likely to churn using a trained machine learning model.
- **High-Risk Customer Identification:** Support prioritization of customers for potential retention campaigns.
- **Feature Engineering:** Convert retail transactions into customer-level predictive features.
- **Model Evaluation:** Assess model performance using classification metrics and ROC-AUC.
- **Temporal Evaluation:** Evaluate predictions using a time-based split to better reflect forecasting on future customers.
- **Interactive Dashboard:** Explore the results through a Streamlit interface with Plotly visualizations.

## Dataset

The project uses the **Online Retail dataset from the UCI Machine Learning Repository**.

The dataset contains retail transaction records, including invoice information, product identifiers, quantities, invoice dates, unit prices, customer identifiers, and countries.

The original transaction data was cleaned and transformed before customer-level features were created.

**Dataset source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

## Project Workflow

1. **Data Collection:** Load the retail transaction dataset.
2. **Data Cleaning:** Handle invalid records, missing values, and unsuitable transactions.
3. **Feature Engineering:** Transform transaction data into customer-level features.
4. **Model Training:** Train machine learning models for churn prediction.
5. **Model Evaluation:** Evaluate performance using classification metrics and temporal testing.
6. **Model Selection:** Use the selected trained model and probability threshold for predictions.
7. **Dashboard Development:** Present customer analytics, high-risk customers, predictions, and evaluation results through Streamlit.

## Machine Learning Approach

The project explores classification models to identify customers who may churn.

### Logistic Regression

Logistic Regression estimates the probability of churn and provides a baseline classification model. The temporal evaluation achieved a ROC-AUC of 0.7426 and recall of 88.84% at the selected threshold of 0.25.

### Random Forest

Random Forest is included as an alternative classification approach that can capture nonlinear relationships between customer features and churn.

### Evaluation Metrics

- **ROC-AUC:** Measures how well the model ranks churned and retained customers across thresholds.
- **Precision:** Measures the proportion of customers predicted to churn who actually churned.
- **Recall:** Measures the proportion of actual churned customers correctly identified.
- **F1-score:** Balances precision and recall.
- **Accuracy:** Measures the proportion of all predictions that are correct.

A temporal evaluation is especially useful because it tests the model on a later period rather than relying only on a random split.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Matplotlib |
| Web Application | Streamlit |
| Model Persistence | Joblib |
| Development Tools | VS Code, Git, GitHub |

## Running the Project Locally

### Prerequisites

- Python 3.12 or another compatible Python version
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/anvitharv-create/Customer-Retention-Intelligence.git
cd Customer-Retention-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the dashboard

```bash
streamlit run app.py
```

The application should open in your browser. If the command fails, check the actual application entry-point filename and the setup instructions in the repository.

## Business Applications

The project demonstrates how machine learning can support:

- Early identification of customers at risk of leaving.
- Prioritization of customer retention campaigns.
- Analysis of purchasing behavior and customer segments.
- Data-driven marketing and customer relationship management.
- Evaluation of predictive models before using them in business workflows.

The model provides predictions, not guarantees of future customer behavior. Business decisions should also consider the cost of retention campaigns and the consequences of incorrect predictions.

## Limitations

- The model learns from historical retail transactions and may not capture every reason a customer stops purchasing.
- Churn labels depend on the observation and prediction periods selected during feature engineering.
- Evaluation results may change with different data splits, features, and classification thresholds.
- The dataset represents a particular retail context, so performance may differ on other businesses or customer populations.

## Future Improvements

- Add model comparison and hyperparameter tuning.
- Improve probability calibration and threshold selection based on business costs.
- Add explainability features to show which factors influence churn predictions.
- Incorporate additional customer demographics or marketing interaction data when available.
- Add retention campaign recommendations and track their effectiveness.


## Author

**Anvitha R V**

- GitHub: https://github.com/anvitharv-create
- LinkedIn: https://linkedin.com/in/anvitharv

---

*Built to demonstrate an end-to-end machine learning workflow, from retail data preprocessing and customer feature engineering to churn prediction, temporal model evaluation, and interactive analytics.*

