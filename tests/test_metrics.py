from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_generated_encounter_data_has_expected_columns():
    df = pd.read_csv(ROOT / "data" / "synthetic_encounters.csv")
    expected = {
        "encounter_id",
        "clinical_text",
        "actual_icd10",
        "predicted_icd10",
        "hcc",
        "risk_category",
        "review_status",
        "model_confidence",
    }
    assert expected.issubset(df.columns)
    assert len(df) >= 500


def test_precision_recall_metrics_are_bounded():
    metrics = pd.read_csv(ROOT / "reports" / "icd10_precision_recall_metrics.csv")
    assert metrics["precision"].between(0, 1).all()
    assert metrics["recall"].between(0, 1).all()
    assert metrics["f1_score"].between(0, 1).all()
