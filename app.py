"""
Streamlit Web Application: Telco Customer Churn Prediction
Loads pre-trained classification models and evaluates test datasets.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Page configuration
st.set_page_config(
    page_title="Telco Churn Evaluator | BITS Pilani",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<style>
    .portal-main-heading {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F2C59;
        margin-bottom: 0.2rem;
    }
    .portal-sub-heading {
        font-size: 0.98rem;
        color: #475569;
        margin-bottom: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Load saved preprocessor and models from disk
@st.cache_resource
def LoadTelcoArtifacts():
    ModelDir = "model"
    TelcoPrep = joblib.load(os.path.join(ModelDir, "telco_preprocessor.joblib"))
    
    TelcoModels = {
        "Logistic Regression": joblib.load(os.path.join(ModelDir, "telco_logreg_model.joblib")),
        "Decision Tree": joblib.load(os.path.join(ModelDir, "telco_dtree_model.joblib")),
        "KNN": joblib.load(os.path.join(ModelDir, "telco_knn_model.joblib")),
        "Naive Bayes": joblib.load(os.path.join(ModelDir, "telco_nb_model.joblib")),
        "Support Vector Machine": joblib.load(os.path.join(ModelDir, "telco_svm_model.joblib")),
    }
    return TelcoPrep, TelcoModels

TelcoPrep, TelcoModels = LoadTelcoArtifacts()

# App title and subtitle
st.markdown('<div class="portal-main-heading">Telco Customer Churn Classifier Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="portal-sub-heading">BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2 | Interactive Evaluation Portal</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Step 1: Test Data Input")
st.sidebar.markdown("Upload test records with input features and the `Churn` target column.")

UploadCsv = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"],
    help="Upload test_data.csv to evaluate model performance."
)

UseDefaultData = st.sidebar.checkbox(
    "Use bundled test_data.csv",
    value=True if UploadCsv is None else False,
    help="Check to load the 1,409 stratified test instances saved during model training."
)

st.sidebar.markdown("---")
st.sidebar.header("Step 2: Model Selection")
SelectedModel = st.sidebar.selectbox(
    "Select Classifier",
    options=[
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
        "All Models (Benchmark Comparison)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Metrics Calculated:**
- Accuracy
- AUC (ROC-AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
""")

# Load test data
EvalDf = None
if UploadCsv is not None:
    try:
        EvalDf = pd.read_csv(UploadCsv)
        st.sidebar.success(f"Loaded: {EvalDf.shape[0]} rows, {EvalDf.shape[1]} columns")
    except Exception as IngestErr:
        st.error(f"Failed to read CSV: {IngestErr}")
elif UseDefaultData and os.path.exists("test_data.csv"):
    EvalDf = pd.read_csv("test_data.csv")
    st.sidebar.info(f"Loaded bundled test_data.csv ({EvalDf.shape[0]} rows, {EvalDf.shape[1]} columns)")
else:
    st.warning("Please upload a test dataset CSV or select the bundled test_data.csv in the sidebar.")
    st.stop()

# Check for target column
TargetKey = "Churn"
if TargetKey not in EvalDf.columns:
    st.error(f"Target column '{TargetKey}' not found in the uploaded data.")
    st.stop()

# Separate features and target
XRaw = EvalDf.drop(columns=[TargetKey])
YRaw = EvalDf[TargetKey]

# Map target to 0 and 1
if YRaw.dtype == object:
    YTrue = YRaw.map({"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}).fillna(0).astype(int)
else:
    YTrue = YRaw.astype(int)

# Transform features using saved preprocessor
try:
    XProc = TelcoPrep.transform(XRaw)
except Exception as TransErr:
    st.error(f"Feature transformation error: {TransErr}")
    st.stop()

# Single-class warning check
UniqueTargetClasses = np.unique(YTrue)
if len(UniqueTargetClasses) < 2:
    st.warning(
        f"Notice: The uploaded test set contains only one target class ({UniqueTargetClasses[0]}). "
        "ROC-AUC score and ROC curves require at least two distinct classes and will be shown as N/A."
    )

# Preview dataset
with st.expander("Preview Test Dataset (First 5 Rows)", expanded=False):
    st.dataframe(EvalDf.head(5), use_container_width=True)
    Col1, Col2, Col3 = st.columns(3)
    Col1.metric("Total Test Instances", len(EvalDf))
    Col2.metric("Non-Churn (Class 0)", int((YTrue == 0).sum()))
    Col3.metric("Churn (Class 1)", int((YTrue == 1).sum()))


def ComputeTelcoMetrics(ModelObj, XFeat, YLabel):
    """Computes all 6 evaluation metrics for a model."""
    YPred = ModelObj.predict(XFeat)

    if hasattr(ModelObj, "predict_proba"):
        YProb = ModelObj.predict_proba(XFeat)[:, 1]
    elif hasattr(ModelObj, "decision_function"):
        YProb = ModelObj.decision_function(XFeat)
    else:
        YProb = YPred

    Acc = accuracy_score(YLabel, YPred)

    # Guard clause: ROC-AUC requires at least two distinct ground truth classes
    if len(np.unique(YLabel)) > 1:
        Auc = roc_auc_score(YLabel, YProb)
    else:
        Auc = np.nan

    Prec = precision_score(YLabel, YPred, zero_division=0)
    Rec = recall_score(YLabel, YPred, zero_division=0)
    F1 = f1_score(YLabel, YPred, zero_division=0)
    Mcc = matthews_corrcoef(YLabel, YPred)

    return {
        "Accuracy": Acc,
        "AUC": Auc,
        "Precision": Prec,
        "Recall": Rec,
        "F1": F1,
        "MCC": Mcc,
        "YPred": YPred,
        "YProb": YProb
    }


# View 1: Comparison table across all models
if SelectedModel == "All Models (Benchmark Comparison)":
    st.subheader("Model Benchmark Comparison")
    st.markdown("Performance of all five models on the test dataset:")

    SummaryRows = []
    RocPlotDict = {}
    HasMultipleClasses = len(np.unique(YTrue)) > 1

    for Name, Estimator in TelcoModels.items():
        Stats = ComputeTelcoMetrics(Estimator, XProc, YTrue)
        SummaryRows.append({
            "ML Model Name": Name,
            "Accuracy": round(Stats["Accuracy"], 4),
            "AUC": round(Stats["AUC"], 4) if not np.isnan(Stats["AUC"]) else "N/A",
            "Precision": round(Stats["Precision"], 4),
            "Recall": round(Stats["Recall"], 4),
            "F1": round(Stats["F1"], 4),
            "MCC": round(Stats["MCC"], 4)
        })
        if HasMultipleClasses:
            FprArr, TprArr, _ = roc_curve(YTrue, Stats["YProb"])
            RocPlotDict[Name] = (FprArr, TprArr, Stats["AUC"])

    SummaryDf = pd.DataFrame(SummaryRows)
    st.dataframe(SummaryDf, use_container_width=True)

    # Multi-model ROC plot
    st.subheader("ROC Curves Comparison")
    if HasMultipleClasses:
        FigRocAll, AxRocAll = plt.subplots(figsize=(8.5, 4.5))
        for Name, (FprArr, TprArr, AucVal) in RocPlotDict.items():
            AxRocAll.plot(FprArr, TprArr, label=f"{Name} (AUC = {AucVal:.4f})", linewidth=2)
        AxRocAll.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5000)")
        AxRocAll.set_xlabel("False Positive Rate")
        AxRocAll.set_ylabel("True Positive Rate")
        AxRocAll.set_title("ROC Curves Across All 5 Classifiers")
        AxRocAll.legend(loc="lower right")
        AxRocAll.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(FigRocAll)
        plt.close(FigRocAll)
    else:
        st.info("ROC Curves cannot be computed because the test dataset contains only a single class.")

else:
    # View 2: Single model view
    ActiveModel = TelcoModels[SelectedModel]
    ModelStats = ComputeTelcoMetrics(ActiveModel, XProc, YTrue)

    st.subheader(f"Evaluation Metrics: {SelectedModel}")

    # 6 metric cards
    AucCardVal = f"{ModelStats['AUC']:.4f}" if not np.isnan(ModelStats['AUC']) else "N/A"
    MCol1, MCol2, MCol3, MCol4, MCol5, MCol6 = st.columns(6)
    MCol1.metric("Accuracy", f"{ModelStats['Accuracy']:.4f}")
    MCol2.metric("AUC Score", AucCardVal)
    MCol3.metric("Precision", f"{ModelStats['Precision']:.4f}")
    MCol4.metric("Recall", f"{ModelStats['Recall']:.4f}")
    MCol5.metric("F1 Score", f"{ModelStats['F1']:.4f}")
    MCol6.metric("MCC Score", f"{ModelStats['MCC']:.4f}")

    st.markdown("---")

    # Confusion matrix and classification report
    VColLeft, VColRight = st.columns(2)

    with VColLeft:
        st.subheader("Confusion Matrix")
        CmMat = confusion_matrix(YTrue, ModelStats["YPred"], labels=[0, 1])
        FigCm, AxCm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            CmMat,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Churn (0)", "Churn (1)"],
            yticklabels=["No Churn (0)", "Churn (1)"],
            ax=AxCm
        )
        AxCm.set_xlabel("Predicted Label")
        AxCm.set_ylabel("True Label")
        AxCm.set_title(f"Confusion Matrix: {SelectedModel}")
        st.pyplot(FigCm)
        plt.close(FigCm)

    with VColRight:
        st.subheader("Classification Report")
        ReportDict = classification_report(
            YTrue,
            ModelStats["YPred"],
            labels=[0, 1],
            target_names=["No Churn (0)", "Churn (1)"],
            output_dict=True,
            zero_division=0
        )
        ReportDf = pd.DataFrame(ReportDict).transpose()
        st.dataframe(ReportDf.style.format(precision=4), use_container_width=True)

        st.subheader("ROC Curve")
        if len(np.unique(YTrue)) > 1:
            FprArr, TprArr, _ = roc_curve(YTrue, ModelStats["YProb"])
            FigRoc, AxRoc = plt.subplots(figsize=(5, 2.7))
            AxRoc.plot(FprArr, TprArr, color="#1E3A8A", lw=2, label=f"ROC (AUC = {ModelStats['AUC']:.4f})")
            AxRoc.plot([0, 1], [0, 1], color="grey", linestyle="--")
            AxRoc.set_xlabel("False Positive Rate")
            AxRoc.set_ylabel("True Positive Rate")
            AxRoc.legend(loc="lower right")
            AxRoc.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(FigRoc)
            plt.close(FigRoc)
        else:
            st.info("ROC Curve cannot be plotted for single-class test data.")

# Footer
st.markdown("---")
st.markdown(
    "<small style='color: grey;'>BITS Pilani WILP M.Tech (AIML) | Machine Learning Assignment 2</small>",
    unsafe_allow_html=True
)
