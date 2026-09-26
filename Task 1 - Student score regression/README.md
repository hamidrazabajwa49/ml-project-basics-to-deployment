# Task 1 — ML Basics: Student Performance Prediction

Predicts a student's final exam score from study hours, attendance, previous
scores, sleep, and other factors using basic regression models.

## Folder structure
```
Task1-StudentPerformance/
├── data/
│   └── student_performance.csv       # generated dataset (800 rows)
├── images/
│   ├── correlation_heatmap.png
│   ├── scatter_relationships.png
│   ├── final_score_distribution.png
│   ├── actual_vs_predicted.png
│   └── model_comparison_r2.png
├── outputs/
│   ├── best_model.joblib             # trained best model
│   ├── scaler.joblib                 # fitted StandardScaler
│   ├── label_encoders.json
│   ├── feature_columns.json
│   ├── metrics.json                  # evaluation metrics for all models
│   ├── model_comparison.csv
│   └── predictions.csv               # test-set predictions
├── notebooks/
│   └── student_performance_prediction.ipynb
├── pipeline.py                       # standalone script that reproduces everything
├── requirements.txt
└── README.md
```

## Dataset
Synthetically generated, but built with realistic relationships: study hours,
attendance %, previous scores, sleep hours, extracurricular participation,
parental education level, and internet access all influence the final score,
plus noise and a few missing values / outliers to mimic real data.

| Column | Description |
|---|---|
| study_hours | Daily study hours (0–12) |
| attendance_percent | Class attendance % (has some missing values) |
| previous_scores | Previous exam score (0–100) |
| sleep_hours | Average sleep hours (has some missing values) |
| extracurricular | Yes/No |
| parental_education | High School / Bachelors / Masters / PhD |
| internet_access | Yes/No |
| final_score | Target variable (0–100) |

## Pipeline
1. **Preprocessing** — median-impute missing numeric values, label-encode
   categoricals, `StandardScaler` on features, 80/20 train/test split.
2. **EDA** — correlation heatmap, scatter plots of key features vs. score,
   score distribution.
3. **Modeling** — Linear Regression, Decision Tree, Random Forest.
4. **Evaluation** — MAE, RMSE, R².

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **4.52** | **5.44** | **0.706** |
| Decision Tree | 5.95 | 7.40 | 0.457 |
| Random Forest | 5.22 | 6.32 | 0.603 |

Linear Regression performed best — expected, since the synthetic target was
generated from a mostly-linear combination of the features.

## How to run
```bash
pip install -r requirements.txt
python pipeline.py                 # regenerates data/images/outputs
jupyter notebook notebooks/student_performance_prediction.ipynb
```
