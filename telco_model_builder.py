"""
Training pipeline for Telco Churn classification models.
Trains 5 classifiers and computes test evaluation metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def RunTelcoPipeline(
    SrcCsvPath: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ModelDir: str = "model",
    TstCsvPath: str = "test_data.csv",
    SeedVal: int = 42
):
    print(f"Reading dataset: {SrcCsvPath}")
    TelcoRawDf = pd.read_csv(SrcCsvPath)

    # Data cleaning
    TelcoCleanDf = TelcoRawDf.copy()

    # Drop customer id column
    if "customerID" in TelcoCleanDf.columns:
        TelcoCleanDf.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges to float and handle blank strings
    TelcoCleanDf["TotalCharges"] = pd.to_numeric(
        TelcoCleanDf["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )
    TelcoCleanDf["TotalCharges"] = TelcoCleanDf["TotalCharges"].fillna(0.0)

    # Convert Churn target to binary (Yes: 1, No: 0)
    TargetCol = "Churn"
    TelcoCleanDf[TargetCol] = TelcoCleanDf[TargetCol].map({"Yes": 1, "No": 0})

    # Separate numerical and categorical columns
    NumCols = ["tenure", "MonthlyCharges", "TotalCharges"]
    CatCols = [Col for Col in TelcoCleanDf.columns if Col not in NumCols and Col != TargetCol]

    print(f"Features: {len(NumCols)} numeric, {len(CatCols)} categorical")

    # 80-20 stratified train-test split
    TelcoTrnDf, TelcoTstDf = train_test_split(
        TelcoCleanDf,
        test_size=0.20,
        random_state=SeedVal,
        stratify=TelcoCleanDf[TargetCol]
    )

    print(f"Train set: {len(TelcoTrnDf)} rows | Test set: {len(TelcoTstDf)} rows")

    # Save test partition to csv for evaluation in streamlit
    TelcoTstDf.to_csv(TstCsvPath, index=False)
    print(f"Saved test data to {TstCsvPath}")

    # Split features and labels
    XTrnRaw = TelcoTrnDf.drop(columns=[TargetCol])
    YTrn = TelcoTrnDf[TargetCol]

    XTstRaw = TelcoTstDf.drop(columns=[TargetCol])
    YTst = TelcoTstDf[TargetCol]

    # Preprocessing: standard scaling for numeric, one-hot encoding for categorical
    TelcoPrep = ColumnTransformer(
        transformers=[
            ("NumScale", StandardScaler(), NumCols),
            ("CatEnc", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CatCols)
        ]
    )

    XTrnProc = TelcoPrep.fit_transform(XTrnRaw)
    XTstProc = TelcoPrep.transform(XTstRaw)

    # Define the 5 classification models
    TelcoClfDict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=SeedVal),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=SeedVal),
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="uniform", metric="minkowski"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=SeedVal)
    }

    # Custom model artifact names
    ModelFileMap = {
        "Logistic Regression": "telco_logreg_model.joblib",
        "Decision Tree": "telco_dtree_model.joblib",
        "KNN": "telco_knn_model.joblib",
        "Naive Bayes": "telco_nb_model.joblib",
        "Support Vector Machine": "telco_svm_model.joblib"
    }

    # Save preprocessor
    os.makedirs(ModelDir, exist_ok=True)
    joblib.dump(TelcoPrep, os.path.join(ModelDir, "telco_preprocessor.joblib"))

    MetricRecords = []

    print("\nTraining models and evaluating on test set...")
    for ModelName, ClfModel in TelcoClfDict.items():
        # Fit model on training set
        ClfModel.fit(XTrnProc, YTrn)

        # Predictions on test set
        YPred = ClfModel.predict(XTstProc)
        YProb = ClfModel.predict_proba(XTstProc)[:, 1]

        # Calculate metrics
        AccVal = accuracy_score(YTst, YPred)
        
        # Guard clause: ROC-AUC requires at least two distinct target classes
        if len(np.unique(YTst)) > 1:
            AucVal = roc_auc_score(YTst, YProb)
        else:
            AucVal = np.nan

        PrecVal = precision_score(YTst, YPred, zero_division=0)
        RecVal = recall_score(YTst, YPred, zero_division=0)
        F1Val = f1_score(YTst, YPred, zero_division=0)
        MccVal = matthews_corrcoef(YTst, YPred)

        MetricRecords.append({
            "ML Model Name": ModelName,
            "Accuracy": round(AccVal, 4),
            "AUC": round(AucVal, 4) if not np.isnan(AucVal) else "N/A",
            "Precision": round(PrecVal, 4),
            "Recall": round(RecVal, 4),
            "F1": round(F1Val, 4),
            "MCC": round(MccVal, 4)
        })

        # Save model
        SavePath = os.path.join(ModelDir, ModelFileMap[ModelName])
        joblib.dump(ClfModel, SavePath)
        print(f"Saved: {SavePath}")

    TelcoSummaryDf = pd.DataFrame(MetricRecords)
    print("\nModel Evaluation Summary:")
    print(TelcoSummaryDf.to_string(index=False))

    TelcoSummaryDf.to_csv(os.path.join(ModelDir, "evaluation_metrics_summary.csv"), index=False)
    return TelcoSummaryDf


if __name__ == "__main__":
    RunTelcoPipeline()
