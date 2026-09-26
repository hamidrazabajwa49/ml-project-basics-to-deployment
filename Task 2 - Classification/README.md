# Task 2 — ML: Classification Project

Binary classification on a real-world medical dataset: predicting whether a
breast tumor is **malignant** or **benign** from cell-nuclei measurements
(the classic Breast Cancer Wisconsin Diagnostic dataset, loaded from
`sklearn.datasets`). Three algorithms are trained and compared.

## Folder structure
```
Task2-Classification/
├── data/
│   └── breast_cancer.csv
├── images/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── mean_radius_boxplot.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── outputs/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── predictions.csv
│   └── classification_report.txt
├── notebooks/
│   └── classification_project.ipynb
├── pipeline.py
├── requirements.txt
└── README.md
```

## Dataset
569 samples, 30 numeric features (radius, texture, perimeter, area,
smoothness, concavity, etc. — mean/error/worst variants), target: malignant
(0) / benign (1). No missing values; classes are moderately imbalanced
(212 malignant vs 357 benign), so precision/recall/F1/ROC-AUC are reported
alongside accuracy.

## Algorithms compared
- **Logistic Regression** (baseline linear model)
- **Random Forest** (200 trees, max_depth=6)
- **SVM** (RBF kernel, probability estimates enabled)

All trained on `StandardScaler`-scaled features, 80/20 stratified split.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.982** | **0.986** | **0.986** | **0.986** | **0.995** |
| Random Forest | 0.965 | 0.959 | 0.986 | 0.972 | 0.994 |
| SVM | 0.974 | 0.972 | 0.986 | 0.979 | 0.996 |

*(exact numbers are also in `outputs/metrics.json`, and may vary slightly on
re-runs of stochastic models)*

Logistic Regression is selected as best model by F1 score. All three
models perform strongly since these features are highly separable by
diagnosis.

## How to run
```bash
pip install -r requirements.txt
python pipeline.py
jupyter notebook notebooks/classification_project.ipynb
```
