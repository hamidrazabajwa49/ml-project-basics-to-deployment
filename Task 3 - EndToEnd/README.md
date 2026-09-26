# Task 3 — ML: End-to-End Prediction System (Loan Approval)

A full pipeline — **data cleaning → EDA → feature engineering → model
training → hyperparameter tuning → evaluation** — plus a simple command-line
interface for making predictions.

## Folder structure
```
Task3-EndToEnd/
├── data/
│   ├── loan_approval_raw.csv      # messy, uncleaned data
│   └── loan_approval_clean.csv    # after cleaning
├── images/
│   ├── loan_status_distribution.png
│   ├── income_loanamount_boxplots.png
│   ├── approval_rate_by_credit_history.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix_best_model.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── outputs/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── label_encoders.json
│   ├── feature_columns.json
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── predictions.csv
│   └── classification_report.txt
├── notebooks/
│   └── loan_approval_end_to_end.ipynb
├── pipeline.py               # full pipeline, regenerates everything
├── predict_cli.py            # <-- the "simple interface" for predictions
├── requirements.txt
└── README.md
```

## Dataset
A synthetic loan-approval dataset (900 rows) deliberately generated **messy**
to make the cleaning step meaningful: inconsistent casing/whitespace in
categorical fields ("male" vs "Male", " No "), `Dependents` given as text
("3+"), missing values in income/loan amount/credit history, and a handful
of extreme income outliers.

## Pipeline stages
1. **Data cleaning** — normalize text casing/whitespace, convert `3+` to a
   number, mode-impute missing categoricals, median-impute missing numerics,
   cap income outliers at the 99th percentile.
2. **EDA** — approval-rate breakdowns by credit history, income, and loan
   amount.
3. **Feature engineering** — total income, log-transforms of skewed monetary
   features, debt-to-income ratio, has-coapplicant flag, label-encoded
   categoricals.
4. **Model training** — Logistic Regression and Gradient Boosting baselines.
5. **Hyperparameter tuning** — `GridSearchCV` (5-fold, scoring=F1) over
   Random Forest's `n_estimators`, `max_depth`, `min_samples_split`.
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC on a held-out
   test set; confusion matrix and feature importance plots.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.606** | **0.582** | **0.655** | **0.616** | **0.640** |
| Gradient Boosting | 0.556 | 0.538 | 0.563 | 0.551 | 0.559 |
| Random Forest (tuned) | 0.556 | 0.542 | 0.517 | 0.529 | 0.611 |

**Honest note on performance:** the target was generated so that credit
history and income only strongly determine the outcome at the extremes —
the large middle segment (good credit, moderate income) is intentionally
close to a coin flip, mirroring how real loan decisions often have a
genuinely ambiguous middle ground. Logistic Regression edges out the
tuned Random Forest here, which is itself a useful lesson: hyperparameter
tuning doesn't automatically beat a simpler baseline, especially on noisy,
weakly-separable data. `outputs/metrics.json` has the exact numbers from
the last run.

## The "simple interface": `predict_cli.py`
Run it interactively:
```bash
python predict_cli.py
```
It will prompt for each field (income, loan amount, credit history, etc.)
and print the prediction with an approval probability.

Or run it non-interactively with flags:
```bash
python predict_cli.py --income 8000 --coapplicant 2000 --loan-amount 150 \
    --term 360 --credit-history 1 --education Graduate --self-employed No \
    --gender Male --married Yes --dependents 0 --property-area Urban
```

## How to run everything
```bash
pip install -r requirements.txt
python pipeline.py                     # regenerates data/images/outputs
jupyter notebook notebooks/loan_approval_end_to_end.ipynb
python predict_cli.py                  # try the prediction interface
```
