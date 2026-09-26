# Task 4 — Advanced ML: Deploy an ML Model

Trains and optimizes a student-performance regression model, deploys it as a
**FastAPI** service with request validation, and connects it to a simple
HTML/JS dashboard. This README doubles as the deployment documentation.

## Folder structure
```
Task4-Deployment/
├── data/
│   └── student_performance.csv
├── images/
│   ├── actual_vs_predicted.png
│   └── baseline_vs_optimized.png
├── outputs/
│   ├── model.joblib                 # copy of the deployed model (reference)
│   ├── scaler.joblib
│   ├── label_encoders.json
│   ├── feature_columns.json
│   └── metrics.json
├── notebooks/
│   └── train_and_optimize.ipynb
├── app/
│   ├── main.py                      # FastAPI service
│   ├── model_artifacts/             # artifacts actually loaded by the API
│   │   ├── model.joblib
│   │   ├── scaler.joblib
│   │   ├── label_encoders.json
│   │   └── feature_columns.json
│   └── static/
│       └── index.html               # dashboard frontend
├── train_and_optimize.py            # standalone training/optimization script
├── requirements.txt
└── README.md
```

## 1. Model training & optimization
`train_and_optimize.py` (mirrored in the notebook):
1. Generates the student-performance dataset (same structure as Task 1).
2. Preprocesses it (label encoding + `StandardScaler`).
3. Trains a **Linear Regression** baseline.
4. Runs **`GridSearchCV`** (5-fold, scoring=R²) over a Random Forest's
   `n_estimators`, `max_depth`, `min_samples_leaf`.
5. Deploys whichever model generalizes better on the held-out test set.

### Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression (deployed)** | **4.19** | **5.29** | **0.658** |
| Random Forest (tuned: `max_depth=10, min_samples_leaf=1, n_estimators=300`) | 4.58 | 5.80 | 0.589 |

Linear Regression is deployed because it generalized better on the test set
— again, tuning a more complex model doesn't guarantee it beats a simpler
baseline, especially when the underlying relationship is close to linear
(as it is here, by construction of the synthetic data).

## 2. API — FastAPI service (`app/main.py`)

### Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns service status, model version, model type |
| POST | `/predict` | Returns a predicted final score (0–100) |
| GET | `/docs` | Auto-generated interactive Swagger UI |
| GET | `/` | Serves the bundled dashboard |

### Input validation
Requests are validated with a **Pydantic** model:
- `study_hours` (0–16), `attendance_percent` (0–100), `previous_scores`
  (0–100), `sleep_hours` (0–14) — numeric, range-checked, must be finite.
- `extracurricular`, `internet_access` — must be exactly `"Yes"` or `"No"`.
- `parental_education` — must be one of `"High School"`, `"Bachelors"`,
  `"Masters"`, `"PhD"`.

Invalid input (out-of-range numbers, unrecognized categories) returns
**HTTP 422** with a machine-readable `detail` list — no partial or silently
wrong predictions are ever returned.

### Run it locally
```bash
pip install -r requirements.txt
python train_and_optimize.py          # (re)creates app/model_artifacts/
cd app
uvicorn main:app --reload --port 8000
```
Then open **http://127.0.0.1:8000/** for the dashboard, or
**http://127.0.0.1:8000/docs** for interactive API docs.

### Example requests
```bash
# Health check
curl http://127.0.0.1:8000/health

# Valid prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours": 6, "attendance_percent": 90, "previous_scores": 75,
    "sleep_hours": 7.5, "extracurricular": "Yes",
    "parental_education": "Masters", "internet_access": "Yes"
  }'
# -> {"predicted_final_score": 79.87, "model_version": "1.0.0", "input_echo": {...}}

# Invalid: study_hours out of range -> HTTP 422
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 999, "attendance_percent": 90, "previous_scores": 75,
       "sleep_hours": 7.5, "extracurricular": "Yes",
       "parental_education": "Masters", "internet_access": "Yes"}'

# Invalid: bad category -> HTTP 422
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 6, "attendance_percent": 90, "previous_scores": 75,
       "sleep_hours": 7.5, "extracurricular": "Maybe",
       "parental_education": "Masters", "internet_access": "Yes"}'
```
All of the above were tested against a locally running instance during
development and behave as documented (valid request → prediction;
out-of-range value and bad category → `422 Unprocessable Entity`).

## 3. Frontend — dashboard (`app/static/index.html`)
A single self-contained HTML/CSS/JS page, served by FastAPI at `/`. It's a
plain form (study hours, attendance, previous scores, sleep, extracurricular,
parental education, internet access) that calls `POST /predict` via `fetch`
and displays the predicted score, or a clear error message if the API call
or validation fails. No build step or framework needed — it's static HTML
served directly by the same FastAPI app, so there's nothing extra to deploy.

## 4. Deployment pipeline summary
1. **Train & optimize** (`train_and_optimize.py` / notebook) → artifacts
   land in `app/model_artifacts/`.
2. **Serve** the model behind `app/main.py` (FastAPI + Pydantic validation +
   CORS enabled for cross-origin frontend use).
3. **Present** via the bundled dashboard, or any other client that can call
   the JSON API (mobile app, another web app, `curl`, Postman, etc.).
4. **Operate**: `/health` gives a cheap liveness/readiness check for a load
   balancer or uptime monitor; `model_version` in every response lets
   clients detect when a new model has been deployed.

### Notes on productionizing further (not implemented here, but worth knowing)
- Add authentication (API key / OAuth) before exposing `/predict` publicly.
- Put a real ASGI server config behind it (e.g. `uvicorn` with multiple
  workers behind `gunicorn`, or a container orchestrator) rather than the
  single-process `--reload` dev server shown above.
- Add structured logging and request/response metrics (e.g. Prometheus) for
  monitoring drift and latency in production.
- Containerize with a `Dockerfile` (not included, but straightforward: copy
  `app/` and `requirements.txt`, `pip install`, `CMD ["uvicorn", "main:app",
  "--host", "0.0.0.0", "--port", "80"]`).

## How to run everything
```bash
pip install -r requirements.txt
python train_and_optimize.py
jupyter notebook notebooks/train_and_optimize.ipynb   # optional, same pipeline
cd app && uvicorn main:app --reload --port 8000       # then open http://127.0.0.1:8000/
```
