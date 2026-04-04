import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Loading data from Kaggle
if not os.path.exists("data/fraud_dataset/train.csv"):
    os.system("pip install kaggle")
    os.system("kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis")
    os.system("unzip healthcare-provider-fraud-detection-analysis.zip -d data")


# Load data
train = pd.read_csv("data/fraud_dataset/train.csv")
inpatient = pd.read_csv("data/fraud_dataset/inpatient.csv")
outpatient = pd.read_csv("data/fraud_dataset/outpatient.csv")
beneficiary = pd.read_csv("data/fraud_dataset/beneficiary.csv")

# Merge
claims = pd.concat([inpatient, outpatient], axis=0)
claims = claims.merge(beneficiary, on="BeneID", how="left")
data = claims.merge(train, on="Provider", how="left")

# Target
data["target"] = data["PotentialFraud"].map({"Yes": 1, "No": 0})

# Feature engineering
data["ClaimStartDt"] = pd.to_datetime(data["ClaimStartDt"])
data["ClaimEndDt"] = pd.to_datetime(data["ClaimEndDt"])
data["ClaimDuration"] = (data["ClaimEndDt"] - data["ClaimStartDt"]).dt.days

diag_cols = [col for col in data.columns if "ClmDiagnosisCode" in col]
proc_cols = [col for col in data.columns if "ClmProcedureCode" in col]

data["NumDiagnosisCodes"] = data[diag_cols].notna().sum(axis=1)
data["NumProcedureCodes"] = data[proc_cols].notna().sum(axis=1)

features = [
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "ClaimDuration",
    "NumDiagnosisCodes",
    "NumProcedureCodes",
    "IPAnnualReimbursementAmt",
    "OPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualDeductibleAmt"
]

X = data[features].fillna(0)
y = data["target"]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "fraud_model.pkl")

print("Fraud model saved.")
