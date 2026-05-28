"""Create synthetic healthcare risk-adjustment data for the demo project."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "medrisk_sentinel.db"
RANDOM_SEED = 42


CONDITIONS = [
    {
        "icd10": "E11.9",
        "hcc": "HCC18",
        "risk_category": "Diabetes",
        "terms": ["type 2 diabetes", "elevated A1c", "metformin", "glucose monitoring"],
        "base_risk": 0.46,
    },
    {
        "icd10": "I50.9",
        "hcc": "HCC85",
        "risk_category": "Heart Failure",
        "terms": ["congestive heart failure", "shortness of breath", "edema", "reduced ejection fraction"],
        "base_risk": 0.72,
    },
    {
        "icd10": "J44.9",
        "hcc": "HCC111",
        "risk_category": "COPD",
        "terms": ["chronic obstructive pulmonary disease", "wheezing", "inhaler therapy", "smoking history"],
        "base_risk": 0.63,
    },
    {
        "icd10": "N18.3",
        "hcc": "HCC138",
        "risk_category": "Chronic Kidney Disease",
        "terms": ["stage 3 chronic kidney disease", "reduced eGFR", "nephrology follow up", "creatinine elevation"],
        "base_risk": 0.67,
    },
    {
        "icd10": "F32.9",
        "hcc": "HCC59",
        "risk_category": "Depression",
        "terms": ["major depressive disorder", "low mood", "sertraline", "behavioral health referral"],
        "base_risk": 0.31,
    },
    {
        "icd10": "E66.9",
        "hcc": "HCC22",
        "risk_category": "Obesity",
        "terms": ["obesity", "high BMI", "nutrition counseling", "weight management"],
        "base_risk": 0.27,
    },
]

NOISE_PHRASES = [
    "annual wellness visit completed",
    "medication list reviewed",
    "patient denies acute chest pain",
    "follow up scheduled in three months",
    "lab panel ordered for monitoring",
    "care gap outreach completed",
]

PAYERS = ["Medicare Advantage", "Commercial", "ACO", "Managed Medicaid"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
SOURCES = ["Clinical Note", "Claim Header", "Coder Review", "NLP Suggestion"]


def make_note(condition: dict, severity: str) -> str:
    if random.random() < 0.14:
        distractor = random.choice([c for c in CONDITIONS if c["icd10"] != condition["icd10"]])
        selected_terms = random.sample(distractor["terms"], k=2)
        selected_terms.append(random.choice(condition["terms"]))
    else:
        selected_terms = random.sample(condition["terms"], k=random.choice([1, 2, 3]))
    if random.random() < 0.34:
        distractor = random.choice([c for c in CONDITIONS if c["icd10"] != condition["icd10"]])
        selected_terms.extend(random.sample(distractor["terms"], k=random.choice([1, 2])))
    if random.random() < 0.16:
        selected_terms.append("history noted but current status requires coder validation")
    noise = random.sample(NOISE_PHRASES, k=2)
    severity_phrase = {
        "Low": "condition appears stable with routine monitoring",
        "Medium": "active management documented during the encounter",
        "High": "significant disease burden and follow up needs documented",
    }[severity]
    return ". ".join(selected_terms + [severity_phrase] + noise) + "."


def choose_severity(base_risk: float) -> str:
    roll = random.random()
    if roll < base_risk - 0.18:
        return "High"
    if roll < base_risk + 0.25:
        return "Medium"
    return "Low"


def build_dataset(rows: int = 850) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    encounters = []
    code_assignments = []
    monthly_metrics = []

    for encounter_id in range(100001, 100001 + rows):
        condition = random.choice(CONDITIONS)
        severity = choose_severity(condition["base_risk"])
        member_id = random.randint(30000, 30980)
        month = pd.Timestamp("2025-01-01") + pd.DateOffset(months=random.randint(0, 11))
        actual_code = condition["icd10"]
        predicted_code = actual_code

        error_roll = random.random()
        review_status = "True Positive"
        if error_roll < 0.09:
            predicted_code = random.choice([c["icd10"] for c in CONDITIONS if c["icd10"] != actual_code])
            review_status = "False Positive"
        elif error_roll < 0.17:
            predicted_code = None
            review_status = "False Negative"

        confidence = round(float(np.clip(np.random.normal(0.82, 0.12), 0.38, 0.99)), 3)
        risk_score = round(float(np.clip(np.random.normal(condition["base_risk"], 0.12), 0.05, 0.96)), 3)

        encounters.append(
            {
                "encounter_id": encounter_id,
                "member_id": member_id,
                "encounter_month": month.strftime("%Y-%m"),
                "payer": random.choice(PAYERS),
                "region": random.choice(REGIONS),
                "source_system": random.choice(SOURCES),
                "clinical_text": make_note(condition, severity),
                "actual_icd10": actual_code,
                "predicted_icd10": predicted_code or "NO_CODE",
                "hcc": condition["hcc"],
                "risk_category": condition["risk_category"],
                "severity": severity,
                "risk_score": risk_score,
                "model_confidence": confidence,
                "review_status": review_status,
            }
        )

        code_assignments.append(
            {
                "encounter_id": encounter_id,
                "actual_icd10": actual_code,
                "predicted_icd10": predicted_code or "NO_CODE",
                "review_status": review_status,
                "requires_audit": int(review_status != "True Positive" or confidence < 0.65),
            }
        )

    df = pd.DataFrame(encounters)
    code_df = pd.DataFrame(code_assignments)

    grouped = df.groupby(["encounter_month", "risk_category"], as_index=False).agg(
        encounters=("encounter_id", "count"),
        avg_risk_score=("risk_score", "mean"),
        avg_confidence=("model_confidence", "mean"),
        audit_rate=("review_status", lambda s: (s != "True Positive").mean()),
    )
    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(3)
    grouped["avg_confidence"] = grouped["avg_confidence"].round(3)
    grouped["audit_rate"] = grouped["audit_rate"].round(3)
    monthly_metrics = grouped
    return df, code_df, monthly_metrics


def write_outputs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    encounters, code_assignments, monthly_metrics = build_dataset()
    encounters.to_csv(DATA_DIR / "synthetic_encounters.csv", index=False)
    code_assignments.to_csv(DATA_DIR / "code_reconciliation.csv", index=False)
    monthly_metrics.to_csv(DATA_DIR / "monthly_risk_metrics.csv", index=False)

    with sqlite3.connect(DB_PATH) as conn:
        encounters.to_sql("encounters", conn, if_exists="replace", index=False)
        code_assignments.to_sql("code_reconciliation", conn, if_exists="replace", index=False)
        monthly_metrics.to_sql("monthly_risk_metrics", conn, if_exists="replace", index=False)

    print(f"Generated {len(encounters):,} encounters at {DATA_DIR}")
    print(f"SQLite database created at {DB_PATH}")


if __name__ == "__main__":
    write_outputs()
