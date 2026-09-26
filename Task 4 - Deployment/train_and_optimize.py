"""
Task 4 - Advanced ML: Deploy an ML Model
Trains and hyperparameter-optimizes a student-performance regression model,
then saves the artifacts consumed by the FastAPI service in app/main.py.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs"
for d in (DATA_DIR, IMG_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

RNG = np.random.default_rng(11)
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# 1. Dataset (same generative structure as Task 1, regenerated independently
#    so this folder is fully self-contained)
# --------------------------------------------------------------------------
N = 800
study_hours = np.clip(RNG.normal(4.5, 2.0, N), 0, 12)
attendance = np.clip(RNG.normal(82, 12, N), 40, 100)
previous_scores = np.clip(RNG.normal(65, 15, N), 20, 100)
sleep_hours = np.clip(RNG.normal(6.8, 1.3, N), 3, 10)
extracurricular = RNG.choice(["Yes", "No"], size=N, p=[0.45, 0.55])
parental_education = RNG.choice(
    ["High School", "Bachelors", "Masters", "PhD"], size=N, p=[0.35, 0.35, 0.22, 0.08]
)
internet_access = RNG.choice(["Yes", "No"], size=N, p=[0.82, 0.18])

parent_edu_bonus = pd.Series(parental_education).map(
    {"High School": 0, "Bachelors": 2, "Masters": 4, "PhD": 5}).values
extra_bonus = pd.Series(extracurricular).map({"Yes": 1.5, "No": 0}).values
internet_bonus = pd.Series(internet_access).map({"Yes": 2, "No": -3}).values
noise = RNG.normal(0, 5, N)

final_score = np.clip(
    2.6 * study_hours + 0.28 * attendance + 0.32 * previous_scores + 1.1 * sleep_hours
    + parent_edu_bonus + extra_bonus + internet_bonus + noise, 0, 100
)

df = pd.DataFrame({
    "study_hours": study_hours.round(2),
    "attendance_percent": attendance.round(1),
    "previous_scores": previous_scores.round(1),
    "sleep_hours": sleep_hours.round(2),
    "extracurricular": extracurricular,
    "parental_education": parental_education,
    "internet_access": internet_access,
    "final_score": final_score.round(1),
})
df.to_csv(DATA_DIR / "student_performance.csv", index=False)

# --------------------------------------------------------------------------
# 2. Preprocess
# --------------------------------------------------------------------------
label_encoders = {}
df_enc = df.copy()
for col in ["extracurricular", "parental_education", "internet_access"]:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

feature_cols = ["study_hours", "attendance_percent", "previous_scores",
                "sleep_hours", "extracurricular", "parental_education", "internet_access"]
X = df_enc[feature_cols]
y = df_enc["final_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 3. Baseline vs hyperparameter-optimized model
# --------------------------------------------------------------------------
baseline = LinearRegression().fit(X_train_scaled, y_train)
baseline_preds = baseline.predict(X_test_scaled)
baseline_metrics = {
    "MAE": float(mean_absolute_error(y_test, baseline_preds)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, baseline_preds))),
    "R2": float(r2_score(y_test, baseline_preds)),
}

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                     scoring="r2", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
best_preds = best_model.predict(X_test_scaled)
best_metrics = {
    "MAE": float(mean_absolute_error(y_test, best_preds)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, best_preds))),
    "R2": float(r2_score(y_test, best_preds)),
}
print("Baseline (Linear Regression):", baseline_metrics)
print("Optimized (Random Forest, tuned):", best_metrics, grid.best_params_)

# choose whichever generalizes better on the held-out test set
final_name, final_model, final_metrics = (
    ("RandomForest_Tuned", best_model, best_metrics) if best_metrics["R2"] >= baseline_metrics["R2"]
    else ("LinearRegression", baseline, baseline_metrics)
)
print("Deploying:", final_name)

# --------------------------------------------------------------------------
# 4. Plots
# --------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_model.predict(X_test_scaled), alpha=0.6, edgecolor="k")
plt.plot([0, 100], [0, 100], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({final_name})")
plt.tight_layout()
plt.savefig(IMG_DIR / "actual_vs_predicted.png", dpi=120)
plt.close()

comp = pd.DataFrame({"LinearRegression": baseline_metrics, f"{final_name}": final_metrics}).T
plt.figure(figsize=(6, 4))
comp["R2"].plot(kind="bar", color="steelblue")
plt.title("Baseline vs Optimized Model (R2)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(IMG_DIR / "baseline_vs_optimized.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------
# 5. Save artifacts for the API
# --------------------------------------------------------------------------
joblib.dump(final_model, OUT_DIR / "model.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "label_encoders.json", "w") as f:
    json.dump(label_encoders, f, indent=2)
with open(OUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "baseline": {"model": "LinearRegression", **baseline_metrics},
        "optimized": {"model": "RandomForest_Tuned", "best_params": grid.best_params_, **best_metrics},
        "deployed_model": final_name,
    }, f, indent=2)

# also copy artifacts into app/ so the FastAPI service has a local, obvious path
APP_MODEL_DIR = ROOT / "app" / "model_artifacts"
APP_MODEL_DIR.mkdir(exist_ok=True, parents=True)
joblib.dump(final_model, APP_MODEL_DIR / "model.joblib")
joblib.dump(scaler, APP_MODEL_DIR / "scaler.joblib")
with open(APP_MODEL_DIR / "label_encoders.json", "w") as f:
    json.dump(label_encoders, f, indent=2)
with open(APP_MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Task 4 training/optimization complete. Artifacts saved to outputs/ and app/model_artifacts/.")
