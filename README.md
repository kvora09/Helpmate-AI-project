This repository now contains an end-to-end **ML attrition modeling project** for early warning signals, including:

- a multivariate training pipeline (not univariate)
- a stochastic polynomial model
- retained-employee scoring population outputs
- a Streamlit dashboard
- a React/Next.js starter frontend with Google Auth
- a notebook for reproducible training flow

## 1) Business framing

- Target label: `voluntary_attrition` (1 = voluntary resignation, 0 = retained)
- Population to score: **retained employees only**
- Objective: identify early risk so HR/business leaders can intervene before attrition occurs.

## 2) Data inputs

### Core highlighted features (7)
1. `age`
2. `tenure_months`
3. `performance_rating`
4. `engagement_score`
5. `manager_feedback_score`
6. `salary_growth_pct`
7. `promotion_wait_months`

### Additional requested multivariate signals
- `gptw_team_score`
- `feedback_resolution_days`
- `ceo_chat_sentiment`
- `same_batch_attrition_rate` (NGTC/GET style cohort trend)
- `posh_cases_last_12m`
- `safety_issues_last_12m`
- optional categorical context like `department_category`, `work_mode_band`

## 3) Model design

The training pipeline uses:

- imputation + scaling
- polynomial feature interactions (`degree=2`)
- `SGDClassifier` (stochastic optimization)
- probability calibration (`CalibratedClassifierCV`)

This aligns with "something along stochastic polynomial regression" while preserving production-ready probabilistic scores.

## 4) Project structure

```text
src/
  data_utils.py            # schema and feature utilities
  generate_sample_data.py  # synthetic dataset generator
  modeling.py              # model pipeline, training, metrics, artifacts
  train.py                 # CLI to train + score retained employees
streamlit_app/
  app.py                   # interactive dashboard for risk analytics
notebooks/
  attrition_model_notebook.ipynb
nextjs-frontend/
  app/                     # Next.js app router pages
  lib/auth.ts              # NextAuth Google provider starter
requirements.txt
```

## 5) Quickstart (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.generate_sample_data
python -m src.train --data data/attrition_dataset.csv --output artifacts
streamlit run streamlit_app/app.py
```

Outputs:
- `artifacts/attrition_model.joblib`
- `artifacts/metrics.json`
- `artifacts/retained_employee_scores.csv`

## 6) Notebook

Open and run:
- `notebooks/attrition_model_notebook.ipynb`

## 7) Next.js starter frontend

```bash
cd nextjs-frontend
npm install
npm run dev
```

Required environment variables for Google auth:

```bash
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
NEXTAUTH_SECRET=...
NEXTAUTH_URL=http://localhost:3000
```

## 8) Production extension suggestions

- Add feature store ingestion from HRIS, GPTW, incident systems.
- Set monthly model retraining + drift monitoring.
- Add fairness slice metrics (gender, location, job band).
- Add intervention outcome tracking (did manager action reduce risk?).

## 9) Notes

- The included sample data is synthetic for demonstration.
- Replace `data/attrition_dataset.csv` with your attached dataset to train on real records.
