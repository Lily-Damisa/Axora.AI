import streamlit as st
import pandas as pd
import joblib
from datetime import date

from claims_process import process_claim

st.set_page_config(page_title="Axora AI - Health Insurance Claims Processing & Fraud Detection System", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# -----------------------
# SESSION STATE
# -----------------------
if "role" not in st.session_state:
    st.session_state.role = None

# -----------------------
# ROLE SELECTION
# -----------------------
if st.session_state.role is None:

    st.markdown(
        """
        <style>


        .main-container {
            text-align: center;
            padding-top: 50px;
        }
        # .card {
        #     border-radius: 12px;
        #     padding: 125px;
        #     background-color: #f5f7fa;
        #     box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        #     text-align: center;
        # }
        .title {
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: gray;
            margin-bottom: 40px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown('<div class="title">Axora AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Smart Insurance Claims Processing & Fraud Detection for Healthcare</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("👩‍⚕️ Healthcare Provider")
        st.write("Submit and process patient claims quickly and efficiently.")

        if st.button("Continue as Provider"):
            st.session_state.role = "provider"

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🛡️ Insurance Admin")
        st.write("Review claims and detect potential fraud patterns.")

        if st.button("Continue as Admin"):
            st.session_state.role = "admin"

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()
# -----------------------
# MAIN APP
# -----------------------
st.title("Axora AI - Claims Processing System")

if st.button("Switch User"):
    st.session_state.role = None
    st.rerun()

st.write(f"Logged in as: **{st.session_state.role.upper()}**")

# -----------------------
# PATIENT + CLAIM INFO
# -----------------------
st.subheader("Patient & Insurance Details")

col1, col2 = st.columns(2)

with col1:
    insurance_id = st.text_input("Insurance ID")
    hmo = st.text_input("HMO / Insurance Provider")
    patient_age = st.number_input("Patient Age", min_value=0, max_value=120)

with col2:
    visit_type = st.selectbox("Visit Type", ["Outpatient", "Inpatient"])
    claim_date = st.date_input("Claim Date", value=date.today())

# -----------------------
# FINANCIAL + MEDICAL
# -----------------------
st.subheader("Claim Details")

col1, col2 = st.columns(2)

with col1:
    claim_amount = st.number_input(
        "Claim Amount",
        help="Total amount requested for reimbursement"
    )

    deductible = st.number_input(
        "Deductible Paid",
        help="Amount paid by the patient before insurance coverage"
    )

    claim_duration = st.number_input(
        "Claim Duration (Days)",
        help="Number of days patient received care"
    )

with col2:
    num_diag = st.number_input(
        "Diagnosis Count",
        help="Number of diagnoses recorded for this claim"
    )

    num_proc = st.number_input(
        "Procedure Count",
        help="Number of procedures performed"
    )

# -----------------------
# SIMPLIFIED EXPLANATIONS
# -----------------------
st.markdown("### Insurance Coverage Info")

col1, col2 = st.columns(2)

with col1:
    ip_reimb = st.number_input(
        "Inpatient Coverage Amount",
        help="Total amount insurance covers yearly for hospital admissions"
    )

    ip_deduct = st.number_input(
        "Inpatient Deductible",
        help="Amount patient must pay before inpatient coverage applies"
    )

with col2:
    op_reimb = st.number_input(
        "Outpatient Coverage Amount",
        help="Total amount insurance covers yearly for clinic visits"
    )

    op_deduct = st.number_input(
        "Outpatient Deductible",
        help="Amount patient must pay before outpatient coverage applies"
    )

# -----------------------
# PROCESS
# -----------------------
if st.button("Process Claim"):

    claim_result = process_claim({
        "InscClaimAmtReimbursed": claim_amount,
        "DeductibleAmtPaid": deductible
    })

    st.subheader("Claim Processing Result")
    st.write(claim_result)

    if st.session_state.role == "admin":

        X = pd.DataFrame([{
            "InscClaimAmtReimbursed": claim_amount,
            "DeductibleAmtPaid": deductible,
            "ClaimDuration": claim_duration,
            "NumDiagnosisCodes": num_diag,
            "NumProcedureCodes": num_proc,
            "IPAnnualReimbursementAmt": ip_reimb,
            "OPAnnualReimbursementAmt": op_reimb,
            "IPAnnualDeductibleAmt": ip_deduct,
            "OPAnnualDeductibleAmt": op_deduct
        }])

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        st.subheader("Fraud Detection")

        if pred == 1:
            st.error(f"Fraud Detected 🚨 ({prob:.2f})")
        else:
            st.success(f"Legitimate Claim ✅ ({prob:.2f})")

    else:
        st.info("Fraud detection is only available to Insurance Admins.")