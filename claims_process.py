import pandas as pd

def process_claim(claim_data):
    """
    Simulates claim processing logic
    """

    amount = claim_data["InscClaimAmtReimbursed"]
    deductible = claim_data["DeductibleAmtPaid"]

    # Basic logic
    approved_amount = max(amount - deductible, 0)

    # Decision rules
    if amount > 50000:
        status = "REQUIRES MANUAL REVIEW"
    elif approved_amount == 0:
        status = "REJECTED"
    else:
        status = "APPROVED"

    return {
        "status": status,
        "approved_amount": approved_amount
    }