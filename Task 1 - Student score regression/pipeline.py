"""
Task 1 - ML Basics: Student Performance Prediction
Generates a synthetic-but-realistic student performance dataset, performs
preprocessing, trains regression models, evaluates them, and saves all
artifacts (data, images, outputs, model).
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs"
for d in (DATA_DIR, IMG_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# 1. Generate synthetic dataset
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
study_hours_outlier_idx = RNG.choice(N, size=15, replace=False)
study_hours[study_hours_outlier_idx] = RNG.uniform(0, 1, 15)  # a few near-zero outliers

parent_edu_bonus = pd.Series(parental_education).map(
    {"High School": 0, "Bachelors": 2, "Masters": 4, "PhD": 5}
).values
extra_bonus = pd.Series(extracurricular).map({"Yes": 1.5, "No": 0}).values
internet_bonus = pd.Series(internet_access).map({"Yes": 2, "No": -3}).values

noise = RNG.normal(0, 5, N)

final_score = (
    2.6 * study_hours
    + 0.28 * attendance
    + 0.32 * previous_scores
    + 1.1 * sleep_hours
    + parent_edu_bonus
    + extra_bonus
    + internet_bonus
    + noise
)
final_score = np.clip(final_score, 0, 100)

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

# introduce a small amount of realistic missingness
for col in ["attendance_percent", "sleep_hours"]:
    miss_idx = RNG.choice(N, size=20, replace=False)
    df.loc[miss_idx, col] = np.nan

df.to_csv(DATA_DIR / "student_performance.csv", index=False)
print("Saved dataset:", df.shape)

# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
df_clean = df.copy()
df_clean["attendance_percent"] = df_clean["attendance_percent"].fillna(df_clean["attendance_percent"].median())
df_clean["sleep_hours"] = df_clean["sleep_hours"].fillna(df_clean["sleep_hours"].median())

label_encoders = {}
for col in ["extracurricular", "parental_education", "internet_access"]:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

feature_cols = [
    "study_hours", "attendance_percent", "previous_scores",
    "sleep_hours", "extracurricular", "parental_education", "internet_access",
]
X = df_clean[feature_cols]
y = df_clean["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 3. EDA plots
# --------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean[feature_cols + ["final_score"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(IMG_DIR / "correlation_heatmap.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(x="study_hours", y="final_score", data=df_clean, ax=axes[0])
axes[0].set_title("Study Hours vs Final Score")
sns.scatterplot(x="attendance_percent", y="final_score", data=df_clean, ax=axes[1])
axes[1].set_title("Attendance vs Final Score")
sns.scatterplot(x="previous_scores", y="final_score", data=df_clean, ax=axes[2])
axes[2].set_title("Previous Scores vs Final Score")
plt.tight_layout()
plt.savefig(IMG_DIR / "scatter_relationships.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df_clean["final_score"], kde=True, bins=25)
plt.title("Distribution of Final Score")
plt.tight_layout()
plt.savefig(IMG_DIR / "final_score_distribution.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------
# 4. Train models
# --------------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    predictions[name] = preds
    results[name] = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }

best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_model_name]
print("Best model:", best_model_name, results[best_model_name])

# --------------------------------------------------------------------------
# 5. Evaluation plots
# --------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions[best_model_name], alpha=0.6, edgecolor="k")
plt.plot([0, 100], [0, 100], "r--")
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.tight_layout()
plt.savefig(IMG_DIR / "actual_vs_predicted.png", dpi=120)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="teal")
    plt.title(f"Feature Importance ({best_model_name})")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "feature_importance.png", dpi=120)
    plt.close()

model_comparison_df = pd.DataFrame(results).T
plt.figure(figsize=(7, 5))
model_comparison_df["R2"].plot(kind="bar", color="slateblue")
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(IMG_DIR / "model_comparison_r2.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------
# 6. Save outputs
# --------------------------------------------------------------------------
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

pred_df = X_test.copy()
pred_df["actual_final_score"] = y_test.values
for name, preds in predictions.items():
    pred_df[f"predicted_{name}"] = preds
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

joblib.dump(best_model, OUT_DIR / "best_model.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "label_encoders.json", "w") as f:
    json.dump(label_encoders, f, indent=2)
with open(OUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

model_comparison_df.to_csv(OUT_DIR / "model_comparison.csv")

print("Task 1 pipeline complete.")
