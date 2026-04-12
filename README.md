# Axora AI

Axora AI is a healthcare claims processing and fraud detection system designed to simulate real-world insurance workflows while identifying potentially fraudulent claims.

The system combines:
- a rule-based claims processing engine  
- and a machine learning-based fraud detection component  

The goal is to improve efficiency, consistency, and transparency in healthcare insurance systems, particularly within emerging markets.

---

## 🚀 Features

### 1. Claims Processing System
- Structured claim input (patient, treatment, insurance details)
- Inpatient / Outpatient classification
- Rule-based validation and decision engine
- Outputs:
  - Approved
  - Rejected
  - Flagged for review

### 2. Fraud Detection System
- Machine learning model trained on healthcare claims data
- Detects suspicious patterns in claims
- Supports risk-based decision making

### 3. Interactive Web App
- Built with Streamlit
- Real-time claim submission and evaluation
- User roles:
  - Healthcare Provider
  - Insurance Admin

---

## 🧠 System Architecture: 
User Input → Data Validation → Claims Processing Engine → Fraud Detection → Output

---

## 📊 Data

The system uses a Medicare-style healthcare claims dataset sourced from Kaggle.

### Data includes:
- Patient demographics
- Claim amounts
- Inpatient / Outpatient indicators
- Diagnosis & procedure codes
- Provider information

---

## ⚙️ Data Preprocessing

- Handling missing values (imputation / removal)
- Date conversion and feature extraction (e.g age)
- Encoding categorical variables
- Normalization of numerical features
- Outlier handling

---

## 🏗️ Claims Processing Logic

The claims engine is rule-based and simulates real insurer decision-making.

### Example logic:
```python
if missing_fields:
    status = "Rejected"

elif claim_amount > coverage_limit:
    status = "Rejected"

elif deductible > 0:
    status = "Partially Approved"

else:
    status = "Approved"

Fraud Detection (Overview)
	•	Feature engineering based on claim patterns
	•	Model training using supervised learning
	•	Evaluation using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score
	•	Cross-validation applied for robustness

⸻

🖥️ Tech Stack
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	Streamlit

⸻

📦 Project Structure
axora-ai/
│── streamlit_app.py
│── claims_processing.py
│── fraud_model.py
│── requirements.txt
│── data/
│── README.md

**LIVE DEMO**
🔗 https://axoraai.streamlit.app/
