from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "medrisk_sentinel.db"
REPORT_DIR = ROOT / "reports"


st.set_page_config(page_title="MedRisk Sentinel AI", page_icon="🩺", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    with sqlite3.connect(DB_PATH) as conn:
        encounters = pd.read_sql_query("SELECT * FROM encounters", conn)
        monthly = pd.read_sql_query("SELECT * FROM monthly_risk_metrics", conn)
    pr_metrics = pd.read_csv(REPORT_DIR / "icd10_precision_recall_metrics.csv")
    with open(REPORT_DIR / "model_metrics.json", "r", encoding="utf-8") as f:
        model_metrics = json.load(f)
    return encounters, monthly, pr_metrics, model_metrics


encounters, monthly, pr_metrics, model_metrics = load_data()

st.title("MedRisk Sentinel AI")
st.caption("NLP-based clinical text classification, ICD-10 reconciliation, and risk-adjustment analytics")

months = sorted(encounters["encounter_month"].unique())
categories = sorted(encounters["risk_category"].unique())

with st.sidebar:
    st.header("Filters")
    selected_months = st.multiselect("Encounter month", months, default=months[-6:])
    selected_categories = st.multiselect("Risk category", categories, default=categories)
    min_confidence = st.slider("Minimum model confidence", 0.0, 1.0, 0.0, 0.05)

filtered = encounters[
    encounters["encounter_month"].isin(selected_months)
    & encounters["risk_category"].isin(selected_categories)
    & (encounters["model_confidence"] >= min_confidence)
]

tp = int((filtered["review_status"] == "True Positive").sum())
fp = int((filtered["review_status"] == "False Positive").sum())
fn = int((filtered["review_status"] == "False Negative").sum())
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
audit_rate = float((filtered["review_status"] != "True Positive").mean()) if len(filtered) else 0

metric_cols = st.columns(6)
metric_cols[0].metric("Encounters", f"{len(filtered):,}")
metric_cols[1].metric("Precision", f"{precision:.1%}")
metric_cols[2].metric("Recall", f"{recall:.1%}")
metric_cols[3].metric("Audit Queue", f"{audit_rate:.1%}")
metric_cols[4].metric("NLP F1", f"{model_metrics['f1_macro']:.1%}")
metric_cols[5].metric("Accuracy", f"{model_metrics['accuracy']:.1%}")

tab_overview, tab_quality, tab_sql, tab_records = st.tabs(
    ["Risk Overview", "Validation", "SQL Metrics", "Encounter Review"]
)

with tab_overview:
    left, right = st.columns([1.1, 0.9])
    category_counts = filtered.groupby("risk_category", as_index=False).size()
    fig = px.bar(
        category_counts,
        x="risk_category",
        y="size",
        color="risk_category",
        title="HCC-Style Risk Category Distribution",
        labels={"size": "Encounters", "risk_category": "Risk category"},
    )
    left.plotly_chart(fig, use_container_width=True)

    trend = (
        filtered.groupby(["encounter_month", "risk_category"], as_index=False)
        .agg(avg_risk_score=("risk_score", "mean"), encounters=("encounter_id", "count"))
        .sort_values("encounter_month")
    )
    fig2 = px.line(
        trend,
        x="encounter_month",
        y="avg_risk_score",
        color="risk_category",
        markers=True,
        title="Average Risk Score Trend",
    )
    right.plotly_chart(fig2, use_container_width=True)

with tab_quality:
    status_counts = filtered.groupby("review_status", as_index=False).size()
    left, right = st.columns(2)
    left.plotly_chart(
        px.pie(status_counts, names="review_status", values="size", title="Code Assignment Review Status"),
        use_container_width=True,
    )
    confidence = filtered.groupby("review_status", as_index=False)["model_confidence"].mean()
    right.plotly_chart(
        px.bar(confidence, x="review_status", y="model_confidence", title="Confidence by Review Outcome"),
        use_container_width=True,
    )

    st.subheader("Audit Queue")
    audit_queue = filtered[
        (filtered["review_status"] != "True Positive") | (filtered["model_confidence"] < 0.65)
    ][
        [
            "encounter_id",
            "encounter_month",
            "risk_category",
            "actual_icd10",
            "predicted_icd10",
            "review_status",
            "model_confidence",
            "clinical_text",
        ]
    ].sort_values(["review_status", "model_confidence"])
    st.dataframe(audit_queue, use_container_width=True, height=360)

with tab_sql:
    st.subheader("ICD-10 Precision, Recall, and F1")
    st.dataframe(pr_metrics, use_container_width=True)
    fig3 = px.scatter(
        pr_metrics,
        x="precision",
        y="recall",
        size="total_encounters",
        color="icd10",
        hover_data=["false_positive", "false_negative", "f1_score"],
        title="Code-Level PR Performance",
        range_x=[0, 1.05],
        range_y=[0, 1.05],
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab_records:
    st.subheader("Clinical Text and Prediction Review")
    review_cols = [
        "encounter_id",
        "payer",
        "region",
        "source_system",
        "risk_category",
        "hcc",
        "actual_icd10",
        "predicted_icd10",
        "review_status",
        "risk_score",
        "model_confidence",
        "clinical_text",
    ]
    st.dataframe(filtered[review_cols].sort_values("model_confidence"), use_container_width=True, height=520)
