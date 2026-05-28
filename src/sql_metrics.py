"""Compute SQL-backed PR, HCC, and reconciliation analytics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "medrisk_sentinel.db"
REPORT_DIR = ROOT / "reports"


QUERY = """
WITH code_metrics AS (
    SELECT
        actual_icd10 AS icd10,
        SUM(CASE WHEN review_status = 'True Positive' THEN 1 ELSE 0 END) AS true_positive,
        SUM(CASE WHEN review_status = 'False Positive' THEN 1 ELSE 0 END) AS false_positive,
        SUM(CASE WHEN review_status = 'False Negative' THEN 1 ELSE 0 END) AS false_negative,
        COUNT(*) AS total_encounters
    FROM encounters
    GROUP BY actual_icd10
)
SELECT
    icd10,
    true_positive,
    false_positive,
    false_negative,
    total_encounters,
    ROUND(1.0 * true_positive / NULLIF(true_positive + false_positive, 0), 3) AS precision,
    ROUND(1.0 * true_positive / NULLIF(true_positive + false_negative, 0), 3) AS recall,
    ROUND(
        2.0 * (
            (1.0 * true_positive / NULLIF(true_positive + false_positive, 0)) *
            (1.0 * true_positive / NULLIF(true_positive + false_negative, 0))
        ) / NULLIF(
            (1.0 * true_positive / NULLIF(true_positive + false_positive, 0)) +
            (1.0 * true_positive / NULLIF(true_positive + false_negative, 0)),
            0
        ),
        3
    ) AS f1_score
FROM code_metrics
ORDER BY f1_score ASC, total_encounters DESC;
"""


AUDIT_QUERY = """
SELECT
    encounter_month,
    risk_category,
    COUNT(*) AS encounters,
    SUM(CASE WHEN review_status = 'False Positive' THEN 1 ELSE 0 END) AS false_positive_count,
    SUM(CASE WHEN review_status = 'False Negative' THEN 1 ELSE 0 END) AS false_negative_count,
    ROUND(AVG(risk_score), 3) AS avg_risk_score,
    ROUND(AVG(model_confidence), 3) AS avg_model_confidence,
    ROUND(AVG(CASE WHEN review_status != 'True Positive' OR model_confidence < 0.65 THEN 1.0 ELSE 0.0 END), 3) AS audit_queue_rate
FROM encounters
GROUP BY encounter_month, risk_category
ORDER BY encounter_month, audit_queue_rate DESC;
"""


def export_sql_metrics() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        pr_metrics = pd.read_sql_query(QUERY, conn)
        audit_metrics = pd.read_sql_query(AUDIT_QUERY, conn)
    pr_metrics.to_csv(REPORT_DIR / "icd10_precision_recall_metrics.csv", index=False)
    audit_metrics.to_csv(REPORT_DIR / "monthly_audit_queue_metrics.csv", index=False)
    print("Precision/recall metrics")
    print(pr_metrics.to_string(index=False))
    print("\nMonthly audit queue sample")
    print(audit_metrics.head(12).to_string(index=False))


if __name__ == "__main__":
    export_sql_metrics()
