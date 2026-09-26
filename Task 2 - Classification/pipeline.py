"""
Task 2 - ML: Classification Project
Real-world dataset: Breast Cancer Wisconsin (Diagnostic) - disease prediction
(benign vs malignant tumor). Compares Logistic Regression, Random Forest, and
SVM, with full evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion
matrix, ROC curves).
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs"
for d in (DATA_DIR, IMG_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# 1. Load real-world dataset
# --------------------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign
df.to_csv(DATA_DIR / "breast_cancer.csv", index=False)
print("Dataset shape:", df.shape)
print("Class balance:\n", df["target"].value_counts())

# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 3. EDA
# --------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x=df["target"].map({0: "Malignant", 1: "Benign"}))
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(IMG_DIR / "class_distribution.png", dpi=120)
plt.close()

top_feats = ["mean radius", "mean texture", "mean perimeter", "mean area",
             "mean smoothness", "mean concavity", "target"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[top_feats].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Selected Features)")
plt.tight_layout()
plt.savefig(IMG_DIR / "correlation_heatmap.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(x=df["target"].map({0: "Malignant", 1: "Benign"}), y=df["mean radius"])
plt.title("Mean Radius by Diagnosis")
plt.tight_layout()
plt.savefig(IMG_DIR / "mean_radius_boxplot.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------
# 4. Train models
# --------------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
proba_dict = {}
pred_dict = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    pred_dict[name] = preds
    proba_dict[name] = proba
    results[name] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

best_model_name = max(results, key=lambda k: results[k]["f1"])
best_model = models[best_model_name]
print("Best model:", best_model_name, results[best_model_name])

# --------------------------------------------------------------------------
# 5. Evaluation plots
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
for ax, (name, preds) in zip(axes, pred_dict.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    ax.set_title(f"{name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(IMG_DIR / "confusion_matrices.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 6))
for name, proba in proba_dict.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "roc_curves.png", dpi=120)
plt.close()

comparison_df = pd.DataFrame(results).T
plt.figure(figsize=(8, 5))
comparison_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar")
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(IMG_DIR / "model_comparison.png", dpi=120)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values().tail(15)
    plt.figure(figsize=(7, 6))
    importances.plot(kind="barh", color="darkorange")
    plt.title(f"Top 15 Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "feature_importance.png", dpi=120)
    plt.close()

# --------------------------------------------------------------------------
# 6. Save outputs
# --------------------------------------------------------------------------
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

comparison_df.to_csv(OUT_DIR / "model_comparison.csv")

pred_out = X_test.copy()
pred_out["actual"] = y_test.values
for name, preds in pred_dict.items():
    pred_out[f"predicted_{name}"] = preds
pred_out.to_csv(OUT_DIR / "predictions.csv", index=False)

with open(OUT_DIR / "classification_report.txt", "w") as f:
    for name, preds in pred_dict.items():
        f.write(f"=== {name} ===\n")
        f.write(classification_report(y_test, preds, target_names=["Malignant", "Benign"]))
        f.write("\n\n")

joblib.dump(best_model, OUT_DIR / "best_model.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")

print("Task 2 pipeline complete.")
