"""Streamlit dashboard for attrition risk monitoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_utils import get_feature_columns

st.set_page_config(page_title="Attrition Early Warning", layout="wide")
st.title("Employee Attrition - Early Detection")
st.caption("Voluntary resignations only. Retained employees are scored for proactive intervention.")

model_path = Path("artifacts/attrition_model.joblib")
score_path = Path("artifacts/retained_employee_scores.csv")

if not model_path.exists() or not score_path.exists():
    st.warning("Train model first: python -m src.generate_sample_data && python -m src.train")
    st.stop()

model = joblib.load(model_path)
scored = pd.read_csv(score_path)

left, right = st.columns(2)
with left:
    st.metric("Retained Employees Scored", f"{len(scored):,}")
    st.metric("High Risk (>=0.65)", int((scored["attrition_risk"] >= 0.65).sum()))
with right:
    st.metric("Average Risk", f"{scored['attrition_risk'].mean():.2f}")
    st.metric("P90 Risk", f"{scored['attrition_risk'].quantile(0.90):.2f}")

fig = px.histogram(scored, x="attrition_risk", nbins=25, title="Retained Population Risk Distribution")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top At-Risk Employees")
show_cols = [c for c in ["employee_id", "department_category", "work_mode_band", "attrition_risk"] if c in scored.columns]
st.dataframe(scored.sort_values("attrition_risk", ascending=False)[show_cols].head(50), use_container_width=True)

st.subheader("Score a custom employee profile")
input_data = {}
example_row = scored.iloc[0].to_dict()
for col in get_feature_columns(scored):
    if col not in scored.columns:
        continue
    if pd.api.types.is_numeric_dtype(scored[col]):
        input_data[col] = st.number_input(col, value=float(example_row[col]))
    else:
        options = sorted(scored[col].dropna().unique().tolist())
        input_data[col] = st.selectbox(col, options, index=0)

if st.button("Predict risk"):
    pred = model.predict_proba(pd.DataFrame([input_data]))[:, 1][0]
    st.success(f"Predicted attrition risk: {pred:.2%}")
