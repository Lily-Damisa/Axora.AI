import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Loading my data
if not os.path.exists("data/fraud_dataset/train.csv"):
    os.system("pip install kaggle")
    os.system("kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis")
    os.system("unzip healthcare-provider-fraud-detection-analysis.zip -d data")

train = pd.read_csv("data/fraud_dataset/train.csv")
inpatient = pd.read_csv("data/fraud_dataset/inpatient.csv")
outpatient = pd.read_csv("data/fraud_dataset/outpatient.csv")
beneficiary = pd.read_csv("data/fraud_dataset/beneficiary.csv")

claims = pd.concat([inpatient, outpatient], axis=0)
claims = claims.merge(beneficiary, on="BeneID", how="left")
data = claims.merge(train, on="Provider", how="left")

# Target
data["target"] = data["PotentialFraud"].map({"Yes": 1, "No": 0})

# Feature Engineering 
data["ClaimStartDt"] = pd.to_datetime(data["ClaimStartDt"])
data["ClaimEndDt"] = pd.to_datetime(data["ClaimEndDt"])
data["ClaimDuration"] = (data["ClaimEndDt"] - data["ClaimStartDt"]).dt.days
data["ClaimPerDay"] = data["InscClaimAmtReimbursed"] / (data["ClaimDuration"] + 1)

diag_cols = [col for col in data.columns if "ClmDiagnosisCode" in col]
proc_cols = [col for col in data.columns if "ClmProcedureCode" in col]

data["NumDiagnosisCodes"] = data[diag_cols].notna().sum(axis=1)
data["NumProcedureCodes"] = data[proc_cols].notna().sum(axis=1)

# Split 
X = data.copy()
y = X.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2nd step of Feature Engineering 
train_df = X_train.copy()
train_df["target"] = y_train

# Provider-level feature
provider_counts = train_df.groupby("Provider")["target"].count()

X_train["ProviderClaimCount"] = X_train["Provider"].map(provider_counts)
X_test["ProviderClaimCount"] = X_test["Provider"].map(provider_counts)

# Patient-level feature
patient_counts = train_df.groupby("BeneID")["target"].count()

X_train["PatientClaimCount"] = X_train["BeneID"].map(patient_counts)
X_test["PatientClaimCount"] = X_test["BeneID"].map(patient_counts)

# Selecting only useful features
features = [
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "ClaimDuration",
    "NumDiagnosisCodes",
    "NumProcedureCodes",
    "IPAnnualReimbursementAmt",
    "OPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualDeductibleAmt",
    "ProviderClaimCount",
    "ClaimPerDay",
    "PatientClaimCount"
]

X_train = X_train[features].fillna(0)
X_test = X_test[features].fillna(0)

# My Model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

print("\nClass Distribution:")
print(y.value_counts(normalize=True))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross Validation 
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)

print("\nCross-Validation F1 Scores:", cv_scores)
print(f"Mean CV F1 Score: {cv_scores.mean():.4f}")

# Saving my model
joblib.dump(pipeline, "fraud_model.pkl")

print("Fraud model saved.")
