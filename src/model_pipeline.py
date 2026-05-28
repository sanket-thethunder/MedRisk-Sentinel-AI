"""Train and evaluate an NLP classifier for clinical risk categories."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_encounters.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def train_model() -> dict:
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["risk_category"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=3500,
                    stop_words="english",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(train_df["clinical_text"], train_df["risk_category"])
    predictions = pipeline.predict(test_df["clinical_text"])

    metrics = {
        "accuracy": round(accuracy_score(test_df["risk_category"], predictions), 4),
        "precision_macro": round(precision_score(test_df["risk_category"], predictions, average="macro"), 4),
        "recall_macro": round(recall_score(test_df["risk_category"], predictions, average="macro"), 4),
        "f1_macro": round(f1_score(test_df["risk_category"], predictions, average="macro"), 4),
    }

    scored = test_df[
        [
            "encounter_id",
            "clinical_text",
            "actual_icd10",
            "predicted_icd10",
            "hcc",
            "risk_category",
            "review_status",
            "risk_score",
            "model_confidence",
        ]
    ].copy()
    scored["predicted_risk_category"] = predictions
    scored["nlp_match"] = scored["risk_category"] == scored["predicted_risk_category"]

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "risk_category_tfidf_logreg.joblib")
    scored.to_csv(REPORT_DIR / "nlp_scored_holdout.csv", index=False)

    with open(REPORT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(test_df["risk_category"], predictions))

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_model()
