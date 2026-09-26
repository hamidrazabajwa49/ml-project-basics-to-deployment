"""
Task 3 - Simple interface for the loan approval predictor.
Run interactively:  python predict_cli.py
Or non-interactively with flags:  python predict_cli.py --income 5000 --coapplicant 0 \
    --loan-amount 120 --term 360 --credit-history 1 --education Graduate \
    --self-employed No --gender Male --married Yes --dependents 0 --property-area Urban
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"

model = joblib.load(OUT_DIR / "best_model.joblib")
scaler = joblib.load(OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "label_encoders.json") as f:
    label_encoders = json.load(f)
with open(OUT_DIR / "feature_columns.json") as f:
    feature_cols = json.load(f)


def encode(col, value):
    mapping = label_encoders[col]
    if value not in mapping:
        raise ValueError(f"'{value}' is not a valid value for {col}. Choices: {list(mapping)}")
    return mapping[value]


def build_features(gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount,
                    loan_term, credit_history, property_area):
    total_income = applicant_income + coapplicant_income
    row = {
        "Gender": encode("Gender", gender),
        "Married": encode("Married", married),
        "Dependents": int(dependents),
        "Education": encode("Education", education),
        "Self_Employed": encode("Self_Employed", self_employed),
        "LogTotalIncome": np.log1p(total_income),
        "LogLoanAmount": np.log1p(loan_amount),
        "Loan_Amount_Term": loan_term,
        "Credit_History": float(credit_history),
        "Property_Area": encode("Property_Area", property_area),
        "DebtToIncomeRatio": loan_amount / (total_income + 1),
        "HasCoapplicant": int(coapplicant_income > 0),
    }
    return pd.DataFrame([row])[feature_cols]


def predict(**kwargs):
    X = build_features(**kwargs)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    return ("Approved" if pred == 1 else "Rejected"), proba


def ask(prompt, choices=None, cast=str):
    while True:
        raw = input(f"{prompt}{' ' + str(choices) if choices else ''}: ").strip()
        try:
            val = cast(raw)
            if choices and val not in choices:
                print(f"Please choose one of {choices}")
                continue
            return val
        except ValueError:
            print("Invalid input, try again.")


def interactive():
    print("=== Loan Approval Predictor ===\n")
    gender = ask("Gender", ["Male", "Female"])
    married = ask("Married", ["Yes", "No"])
    dependents = ask("Number of dependents", ["0", "1", "2", "3"])
    education = ask("Education", ["Graduate", "Not Graduate"])
    self_employed = ask("Self employed", ["Yes", "No"])
    applicant_income = ask("Applicant monthly income", cast=float)
    coapplicant_income = ask("Coapplicant monthly income (0 if none)", cast=float)
    loan_amount = ask("Loan amount requested (in thousands)", cast=float)
    loan_term = ask("Loan term in months", cast=float)
    credit_history = ask("Credit history meets guidelines? (1 = yes, 0 = no)", ["0", "1"])
    property_area = ask("Property area", ["Urban", "Semiurban", "Rural"])

    label, proba = predict(
        gender=gender, married=married, dependents=dependents, education=education,
        self_employed=self_employed, applicant_income=applicant_income,
        coapplicant_income=coapplicant_income, loan_amount=loan_amount,
        loan_term=loan_term, credit_history=credit_history, property_area=property_area,
    )
    print(f"\nPrediction: {label}  (approval probability: {proba:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Loan approval predictor")
    parser.add_argument("--income", type=float)
    parser.add_argument("--coapplicant", type=float)
    parser.add_argument("--loan-amount", type=float)
    parser.add_argument("--term", type=float, default=360)
    parser.add_argument("--credit-history", type=int, choices=[0, 1])
    parser.add_argument("--education", choices=["Graduate", "Not Graduate"])
    parser.add_argument("--self-employed", choices=["Yes", "No"])
    parser.add_argument("--gender", choices=["Male", "Female"])
    parser.add_argument("--married", choices=["Yes", "No"])
    parser.add_argument("--dependents", choices=["0", "1", "2", "3"])
    parser.add_argument("--property-area", choices=["Urban", "Semiurban", "Rural"])
    args = parser.parse_args()

    if args.income is None:
        interactive()
        return

    label, proba = predict(
        gender=args.gender, married=args.married, dependents=args.dependents,
        education=args.education, self_employed=args.self_employed,
        applicant_income=args.income, coapplicant_income=args.coapplicant or 0,
        loan_amount=args.loan_amount, loan_term=args.term,
        credit_history=args.credit_history, property_area=args.property_area,
    )
    print(f"Prediction: {label}  (approval probability: {proba:.1%})")


if __name__ == "__main__":
    main()
