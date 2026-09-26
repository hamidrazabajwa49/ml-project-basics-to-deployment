"""
Task 3 - ML: End-to-End Prediction System (Loan Approval)
Data cleaning -> EDA -> feature engineering -> model training ->
hyperparameter tuning -> evaluation. Artifacts are saved so that
predict_cli.py can load the final model and serve predictions.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs"
for d in (DATA_DIR, IMG_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

RNG = np.random.default_rng(7)
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# 1. Generate a realistically MESSY loan-approval dataset
# --------------------------------------------------------------------------
N = 900

gender = RNG.choice(["Male", "Female", "male", "FEMALE"], size=N, p=[0.45, 0.35, 0.1, 0.1])
married = RNG.choice(["Yes", "No", "yes", " No "], size=N, p=[0.5, 0.3, 0.1, 0.1])
dependents = RNG.choice(["0", "1", "2", "3+"], size=N, p=[0.5, 0.2, 0.2, 0.1])
education = RNG.choice(["Graduate", "Not Graduate"], size=N, p=[0.7, 0.3])
self_employed = RNG.choice(["Yes", "No", np.nan], size=N, p=[0.15, 0.75, 0.10])
property_area = RNG.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.4, 0.35, 0.25])
credit_history = RNG.choice([1.0, 0.0, np.nan], size=N, p=[0.78, 0.12, 0.10])

applicant_income = np.round(np.clip(RNG.lognormal(8.5, 0.5, N), 1000, 40000))
coapplicant_income = np.round(np.clip(RNG.lognormal(7.0, 1.2, N) - 500, 0, 20000))
coapplicant_income[RNG.random(N) < 0.35] = 0  # many single applicants

loan_amount = np.round(np.clip(
    (applicant_income + coapplicant_income) * RNG.uniform(0.05, 0.25, N) / 10, 20, 700
))
loan_amount_term = RNG.choice([360, 180, 120, 84, 60], size=N, p=[0.7, 0.1, 0.1, 0.05, 0.05]).astype(float)

# inject missing values and outliers to simulate messy real-world data
for arr, frac in [(applicant_income, 0.02), (loan_amount, 0.03), (loan_amount_term, 0.02)]:
    idx = RNG.choice(N, size=int(N * frac), replace=False)
    arr[idx] = np.nan
outlier_idx = RNG.choice(N, size=8, replace=False)
applicant_income[outlier_idx] = applicant_income[outlier_idx] * 8  # extreme outliers

total_income_true = np.nan_to_num(applicant_income) + coapplicant_income
dep_numeric = pd.Series(dependents).str.replace("3+", "3", regex=False).astype(float).values
credit_hist_filled = np.nan_to_num(credit_history, nan=0.5)

self_emp_penalty = pd.Series(self_employed).map({"Yes": -0.2, "No": 0.1}).fillna(0).values
score = (
    0.00018 * total_income_true
    + 4.0 * credit_hist_filled
    + pd.Series(education).map({"Graduate": 0.6, "Not Graduate": -0.3}).values
    - 0.1 * dep_numeric
    + self_emp_penalty
    + RNG.normal(0, 0.6, N)
)
approval_prob = 1 / (1 + np.exp(-(score - score.mean()) / score.std()))
loan_status = (RNG.random(N) < approval_prob).astype(int)  # 1 = Approved, 0 = Rejected

df = pd.DataFrame({
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": np.where(loan_status == 1, "Y", "N"),
})
df.to_csv(DATA_DIR / "loan_approval_raw.csv", index=False)
print("Raw (messy) dataset saved:", df.shape)
print("Missing values:\n", df.isna().sum())

# --------------------------------------------------------------------------
# 2. Data cleaning
# --------------------------------------------------------------------------
clean = df.copy()
clean["Gender"] = clean["Gender"].str.strip().str.title()
clean["Married"] = clean["Married"].str.strip().str.title()
clean["Dependents"] = clean["Dependents"].str.replace("3+", "3", regex=False)

for col in ["Gender", "Married", "Self_Employed", "Dependents", "Credit_History", "Loan_Amount_Term"]:
    mode_val = clean[col].mode(dropna=True)[0]
    clean[col] = clean[col].fillna(mode_val)

clean["Dependents"] = clean["Dependents"].astype(int)
clean["Credit_History"] = clean["Credit_History"].astype(float)

clean["ApplicantIncome"] = clean["ApplicantIncome"].fillna(clean["ApplicantIncome"].median())
income_cap = clean["ApplicantIncome"].quantile(0.99)
clean["ApplicantIncome"] = np.clip(clean["ApplicantIncome"], None, income_cap)  # cap outliers

clean["LoanAmount"] = clean["LoanAmount"].fillna(clean["LoanAmount"].median())

clean.to_csv(DATA_DIR / "loan_approval_clean.csv", index=False)
print("\nAfter cleaning, missing values:\n", clean.isna().sum())

# --------------------------------------------------------------------------
# 3. EDA
# --------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x="Loan_Status", data=clean)
plt.title("Loan Approval Distribution")
plt.tight_layout()
plt.savefig(IMG_DIR / "loan_status_distribution.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=clean, ax=axes[0])
axes[0].set_title("Applicant Income by Loan Status (outlier-capped)")
sns.boxplot(x="Loan_Status", y="LoanAmount", data=clean, ax=axes[1])
axes[1].set_title("Loan Amount by Loan Status")
plt.tight_layout()
plt.savefig(IMG_DIR / "income_loanamount_boxplots.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
sns.barplot(x="Credit_History", y=(clean["Loan_Status"] == "Y").astype(int), data=clean, errorbar=None)
plt.ylabel("Approval Rate")
plt.title("Approval Rate by Credit History")
plt.tight_layout()
plt.savefig(IMG_DIR / "approval_rate_by_credit_history.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------------------------------
feat = clean.copy()
feat["TotalIncome"] = feat["ApplicantIncome"] + feat["CoapplicantIncome"]
feat["LogTotalIncome"] = np.log1p(feat["TotalIncome"])
feat["LogLoanAmount"] = np.log1p(feat["LoanAmount"])
feat["DebtToIncomeRatio"] = feat["LoanAmount"] / (feat["TotalIncome"] + 1)
feat["HasCoapplicant"] = (feat["CoapplicantIncome"] > 0).astype(int)

label_encoders = {}
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
    le = LabelEncoder()
    feat[col] = le.fit_transform(feat[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

feat["Loan_Status"] = feat["Loan_Status"].map({"N": 0, "Y": 1})

feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "LogTotalIncome", "LogLoanAmount", "Loan_Amount_Term", "Credit_History",
    "Property_Area", "DebtToIncomeRatio", "HasCoapplicant",
]
X = feat[feature_cols]
y = feat["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(9, 7))
sns.heatmap(feat[feature_cols + ["Loan_Status"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap (after engineering)")
plt.tight_layout()
plt.savefig(IMG_DIR / "correlation_heatmap.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------
# 5. Model training + hyperparameter tuning
# --------------------------------------------------------------------------
baseline_models = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}
baseline_results = {}
for name, model in baseline_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    baseline_results[name] = float(f1_score(y_test, preds))
print("Baseline F1 scores:", baseline_results)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="f1", n_jobs=-1
)
grid.fit(X_train_scaled, y_train)
best_rf = grid.best_estimator_
print("Best RF params:", grid.best_params_)

# --------------------------------------------------------------------------
# 6. Final evaluation
# --------------------------------------------------------------------------
final_candidates = {**baseline_models, "RandomForest_Tuned": best_rf}
final_results = {}
final_preds = {}
final_proba = {}
for name, model in final_candidates.items():
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    final_preds[name] = preds
    final_proba[name] = proba
    final_results[name] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

best_model_name = max(final_results, key=lambda k: final_results[k]["f1"])
best_model = final_candidates[best_model_name]
print("Final best model:", best_model_name, final_results[best_model_name])

plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, final_preds[best_model_name])
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(IMG_DIR / "confusion_matrix_best_model.png", dpi=120)
plt.close()

comp_df = pd.DataFrame(final_results).T
plt.figure(figsize=(8, 5))
comp_df.plot(kind="bar")
plt.title("Model Comparison (baseline vs. tuned)")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(IMG_DIR / "model_comparison.png", dpi=120)
plt.close()

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(7, 6))
    imp.plot(kind="barh", color="seagreen")
    plt.title(f"Feature Importance ({best_model_name})")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "feature_importance.png", dpi=120)
    plt.close()

# --------------------------------------------------------------------------
# 7. Save all artifacts (used by predict_cli.py)
# --------------------------------------------------------------------------
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({"baseline_f1": baseline_results, "final_results": final_results,
               "best_model": best_model_name, "best_rf_params": grid.best_params_}, f, indent=2)

comp_df.to_csv(OUT_DIR / "model_comparison.csv")

with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(classification_report(y_test, final_preds[best_model_name], target_names=["Rejected", "Approved"]))

pred_out = X_test.copy()
pred_out["actual"] = y_test.values
pred_out["predicted"] = final_preds[best_model_name]
pred_out.to_csv(OUT_DIR / "predictions.csv", index=False)

joblib.dump(best_model, OUT_DIR / "best_model.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "label_encoders.json", "w") as f:
    json.dump(label_encoders, f, indent=2)
with open(OUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Task 3 pipeline complete.")
