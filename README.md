# BITS Pilani — M.Tech (AIML) Machine Learning: Assignment 2

## a. Problem Statement
Customer churn is an important challenge for telecom service providers. Acquiring new subscribers requires significantly more resources than retaining existing customers. The objective of this project is to build an end-to-end machine learning classification pipeline to predict whether a customer will churn (`Churn = Yes`) or stay (`Churn = No`).

By identifying subscribers at risk of churn based on their account tenure, service usage, and contract details, telecom companies can proactively offer targeted retention incentives (such as tailored loyalty discounts or technical assistance) to reduce customer loss.

---

## b. Dataset Description
The analysis uses the **Telco Customer Churn Dataset** (sourced from IBM / Kaggle).

- **Total Instances**: 7,043 customer accounts
- **Total Features**: 19 predictor attributes and 1 binary target column
- **Target Variable**: `Churn` (Binary: `Yes` = 1, `No` = 0)
- **Target Distribution**: Approximately 73.4% non-churn (`No`) and 26.6% churn (`Yes`).

### Features Breakdown:
1. **Numerical Features (3)**:
   - `tenure`: Number of months the subscriber has stayed with the company.
   - `MonthlyCharges`: Current monthly fee billed to the customer.
   - `TotalCharges`: Cumulative charges over the customer's lifespan (blank entries for new accounts with tenure 0 were imputed to 0.0).
2. **Categorical Features (16)**:
   - *Customer Demographics*: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
   - *Phone Services*: `PhoneService`, `MultipleLines`.
   - *Internet & Security Services*: `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
   - *Billing & Contract Details*: `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod` (Electronic check, Mailed check, Bank transfer, Credit card).

### Data Preprocessing & Validation:
- The unique identifier `customerID` was removed before modeling to avoid data leakage.
- The dataset was split into 80% training (5,634 rows) and 20% testing (1,409 rows) using stratified sampling (`random_state=42`) to maintain class balance across splits.
- A `ColumnTransformer` was fitted strictly on the training partition:
  - Continuous variables were scaled using `StandardScaler`.
  - Categorical variables were encoded using `OneHotEncoder(drop='first', sparse_output=False)`.
- The 1,409 test records were saved as `test_data.csv` with the `Churn` target column preserved so the Streamlit application can load and evaluate test data without retraining.

---

## c. GitHub Repository Link
**Repository URL**: `https://github.com/Janani3697/telco-customer-churn-ml`

### Repository Layout:
```text
project-folder/
│-- .gitignore                          # Standard git ignore file
│-- app.py                              # Streamlit web application
│-- requirements.txt                    # Python package dependencies
│-- README.md                           # Documentation and evaluation report
│-- test_data.csv                       # Unseen test dataset (1,409 rows)
│-- telco_model_builder.py              # Model training and artifact generation script
│-- telco_churn_lab.ipynb               # Jupyter notebook executed on BITS Virtual Lab
│-- WA_Fn-UseC_-Telco-Customer-Churn.csv # Full dataset
└── model/                              # Serialized model artifacts
    │-- telco_logreg_model.joblib       # Logistic Regression model
    │-- telco_dtree_model.joblib        # Decision Tree model
    │-- telco_knn_model.joblib          # KNN model
    │-- telco_nb_model.joblib           # Naive Bayes model
    │-- telco_svm_model.joblib          # Support Vector Machine model
    │-- telco_preprocessor.joblib       # Fitted ColumnTransformer preprocessor
    │-- evaluation_metrics_summary.csv  # Benchmark metrics table
    │-- telco_model_builder.py          # Source training script
    └── telco_churn_lab.ipynb           # Source notebook
```

---

## d. Models Used & Evaluation Metrics Comparison

Five classification models were trained on the training partition and evaluated on the identical 1,409 test records:

### Evaluation Metrics Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8062** | **0.8422** | 0.6593 | 0.5588 | **0.6049** | **0.4806** |
| **Decision Tree** | 0.8013 | 0.8281 | **0.6679** | 0.5000 | 0.5719 | 0.4539 |
| **KNN** | 0.7729 | 0.8084 | 0.5730 | 0.5668 | 0.5699 | 0.4156 |
| **Naive Bayes** | 0.6558 | 0.8092 | 0.4269 | **0.8663** | 0.5719 | 0.3951 |
| **Support Vector Machine** | 0.7949 | 0.7948 | 0.6545 | 0.4813 | 0.5547 | 0.4339 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers the best overall performance across most metrics (**Accuracy: 0.8062, AUC: 0.8422, F1: 0.6049, MCC: 0.4806**). Churn drivers like tenure length, contract type, and monthly charges exhibit a relatively monotonic relationship with the log-odds of churning, which the linear decision boundary captures effectively without overfitting. It maintains a solid balance between precision (0.6593) and recall (0.5588). |
| **Decision Tree** | Performs competitively (**Accuracy: 0.8013, AUC: 0.8281, Precision: 0.6679**). With regularization (`max_depth=6`, `min_samples_split=20`), the tree isolates clean decision splits (such as month-to-month contracts combined with high monthly charges). It produces the highest precision among the five models (0.6679), though its recall (0.5000) shows that leaves are relatively conservative. |
| **KNN** | Achieves balanced precision (0.5730) and recall (0.5668) with an **Accuracy of 0.7729** and **AUC of 0.8084**. Because one-hot encoding expands the feature space to over 30 dimensions, Euclidean distance differences between neighbors become less distinct (curse of dimensionality), resulting in an MCC score of 0.4156. |
| **Naive Bayes** | Shows an interesting trade-off: the highest **Recall at 0.8663** (detecting 86.6% of actual churners), but the lowest **Precision (0.4269)** and **Accuracy (0.6558)**. This is caused by the violation of feature independence. In telecom plans, bundled services (like internet service, online backup, device protection, streaming TV, and higher monthly charges) are correlated. Correlated evidence causes Naive Bayes to overstate posterior churn probabilities, creating more false positives. |
| **Support Vector Machine** | Yields reasonable accuracy (**0.7949**) and precision (**0.6545**), but lower recall (**0.4813**) and **MCC (0.4339)**. With the RBF kernel, margin optimization on the imbalanced 73:27 class distribution slightly biases the boundary toward the majority class, leading to a conservative prediction of positive churn instances. |
| **Overall, Winner for your dataset?** | **Logistic Regression is the overall winner.** It leads in Accuracy (**0.8062**), discriminatory capability across thresholds (**AUC = 0.8422**), balanced F1-score (**0.6049**), and Matthews Correlation Coefficient (**MCC = 0.4806**). In customer retention, where both false alarms and missed defections incur real costs, Logistic Regression provides the most calibrated and balanced performance. |

---

## e. Streamlit Web Application
- **Platform**: Streamlit Community Cloud
- **Live App URL**: `https://telco-customer-churn-ml-kfv6xiiez9wuxizqgvkcad.streamlit.app/`  

### Application Overview:
1. **Test Data Ingestion**: Supports uploading any test CSV containing the 19 features and target column, with an option to load the bundled 1,409 test instances.
2. **Model Selection**: Switch between individual classifiers or view the comparison table and ROC curves across all models simultaneously.
3. **Metrics Display**: Renders real-time cards for Accuracy, ROC-AUC, Precision, Recall, F1, and MCC.
4. **Visualizations**: Displays an interactive Confusion Matrix heatmap, a formatted Classification Report, and ROC curves.
5. **No Retraining**: All predictions and metric calculations run instantly by passing test records through the saved preprocessor and model `.joblib` files.
