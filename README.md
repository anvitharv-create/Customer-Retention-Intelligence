# Customer Retention Intelligence

An end-to-end machine learning project that analyzes customer purchasing behavior, predicts customer churn, and helps businesses identify customers who may stop purchasing.

## Overview

Customer retention is essential for sustainable business growth. Losing existing customers can reduce revenue and increase the cost of acquiring new customers.

**Customer Retention Intelligence** transforms historical retail transaction data into customer-level features and uses machine learning to predict potential churn. The project combines data preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit dashboard to turn transaction data into actionable insights.

The dashboard enables users to explore customer behavior, identify high-risk customers, generate churn predictions, and review model performance.

## Live Demo

- **GitHub Repository:** [Customer Retention Intelligence](https://github.com/anvitharv-create/Customer-Retention-Intelligence)
- **Live Dashboard:** [Open Streamlit App](https://customer-retention-intelligence-69jqdy8zdbwmacrgmtvk7d.streamlit.app)

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

This project addresses the problem by transforming transaction-level data into customer-level features and applying machine learning to identify potential churn. These insights can help businesses prioritize retention efforts and make more informed decisions.

## Key Results

| Metric | Result |
|---|---:|
| Cleaned transaction records | 392,692 |
| Customers in the feature dataset | 3,370 |
| Customers labelled as churned | 43.06% |
| Temporal test ROC-AUC | 0.7426 |
| Temporal test recall | 88.84% |
| Temporal test F1-score | 68.18% |

*The temporal evaluation results shown above are from Logistic Regression at a probability threshold of 0.25. These results belong to the temporal evaluation workflow and should not be confused with the separate random-split evaluation.*

## Key Features

- **Customer Analytics:** Explore customer-level data and retention patterns.
- **Churn Prediction:** Estimate whether a customer is likely to churn using a trained machine learning model.
- **High-Risk Customer Identification:** Help prioritize customers for potential retention campaigns.
- **Feature Engineering:** Transform retail transactions into customer-level predictive features.
- **Model Comparison:** Compare Logistic Regression and Random Forest using validation metrics.
- **Threshold Optimization:** Select a classification threshold based on validation F1-score.
- **Model Evaluation:** Assess classification performance using ROC-AUC, precision, recall, F1-score, accuracy, and PR-AUC.
- **Temporal Evaluation:** Evaluate model performance using a separate time-based evaluation workflow.
- **Interactive Dashboard:** Explore analytics, predictions, high-risk customers, and model performance through Streamlit and Plotly.

## Dataset

The project uses the **Online Retail dataset** from the UCI Machine Learning Repository.

The dataset contains retail transaction records, including invoice information, product identifiers, quantities, invoice dates, unit prices, customer identifiers, and countries.

The original transaction data is cleaned and transformed before customer-level features are generated.

**Dataset source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

## Project Workflow

1. **Data Collection:** Load the retail transaction dataset.
2. **Data Cleaning:** Handle invalid records, missing values, and unsuitable transactions.
3. **Feature Engineering:** Transform transaction data into customer-level predictive features.
4. **Model Training:** Train Logistic Regression and Random Forest classifiers.
5. **Model Comparison:** Compare the models using validation ROC-AUC.
6. **Threshold Selection:** Select the decision threshold that achieves the highest validation F1-score among the tested thresholds.
7. **Test Evaluation:** Evaluate the selected model on a separate test set.
8. **Temporal Evaluation:** Assess performance on a later time period through the separate temporal evaluation workflow.
9. **Dashboard Development:** Present customer analytics, high-risk customers, predictions, and model performance through Streamlit.

## Machine Learning Approach

The project implements two classification models to predict customer churn.

### Logistic Regression

Logistic Regression estimates the probability of customer churn and provides a classification baseline. The training pipeline uses median imputation, feature scaling, and class weighting.

The separate temporal evaluation achieved a ROC-AUC of **0.7426** and recall of **88.84%** at a selected probability threshold of **0.25**.

### Random Forest

Random Forest uses an ensemble of decision trees to capture nonlinear relationships between customer features and churn. Its training pipeline uses median imputation and class weighting.

### Model Selection and Threshold Optimization

The training script fits both models and compares their validation performance. The model with the highest validation ROC-AUC is selected.

The script then tests multiple probability thresholds and selects the threshold with the highest validation F1-score. The selected model and threshold are evaluated on a separate test dataset.

The random-split test results and temporal evaluation results are distinct evaluations and should be interpreted separately.

### Evaluation Metrics

- **ROC-AUC:** Measures the model's ability to distinguish churned and retained customers across classification thresholds.
- **Precision:** Measures the proportion of customers predicted to churn who actually churned.
- **Recall:** Measures the proportion of actual churned customers correctly identified.
- **F1-score:** Balances precision and recall.
- **Accuracy:** Measures the proportion of all predictions that are correct.
- **PR-AUC:** Summarizes precision-recall performance across classification thresholds.

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

On Windows PowerShell:

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

The application should open in your browser. The project also contains scripts in the `src/` directory for data preparation, feature engineering, model training, evaluation, and analysis.

**Note:** Running the training scripts requires the processed customer feature dataset at the expected project path. Follow the data preparation workflow before training the model if that file is not already available.

## Business Applications

The project demonstrates how machine learning can support:

- Early identification of customers at risk of leaving.
- Prioritization of customer retention campaigns.
- Analysis of purchasing behavior and customer segments.
- Data-driven marketing and customer relationship management.
- Evaluation of predictive models before using them in business workflows.

Model predictions are estimates, not guarantees of future customer behavior. Business decisions should also consider the cost of retention campaigns and the consequences of incorrect predictions.

## Limitations

- The model learns from historical retail transactions and may not capture every reason a customer stops purchasing.
- Churn labels depend on the observation and prediction periods defined during feature engineering.
- Evaluation results can change with different data splits, features, and classification thresholds.
- The dataset represents a particular retail context, so performance may differ across businesses and customer populations.
- High recall does not necessarily mean high precision; the two metrics should be considered together when planning retention campaigns.

## Future Improvements

- Add further model comparison and hyperparameter tuning.
- Improve probability calibration and threshold selection based on business costs.
- Add explainability features to show which factors influence churn predictions.
- Incorporate additional customer demographics or marketing interaction data when available.
- Develop retention campaign recommendations and evaluate their effectiveness.

## Author

**Anvitha R V**

- **GitHub:** [anvitharv-create](https://github.com/anvitharv-create)
- **LinkedIn:** [anvitharv](https://linkedin.com/in/anvitharv)

---

*Built to demonstrate an end-to-end machine learning workflow, from retail data preprocessing and customer feature engineering to churn prediction, model evaluation, and interactive analytics.*
