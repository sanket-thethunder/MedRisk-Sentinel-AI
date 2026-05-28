CREATE TABLE encounters (
    encounter_id INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL,
    encounter_month TEXT NOT NULL,
    payer TEXT NOT NULL,
    region TEXT NOT NULL,
    source_system TEXT NOT NULL,
    clinical_text TEXT NOT NULL,
    actual_icd10 TEXT NOT NULL,
    predicted_icd10 TEXT NOT NULL,
    hcc TEXT NOT NULL,
    risk_category TEXT NOT NULL,
    severity TEXT NOT NULL,
    risk_score REAL NOT NULL,
    model_confidence REAL NOT NULL,
    review_status TEXT NOT NULL
);

CREATE TABLE code_reconciliation (
    encounter_id INTEGER NOT NULL,
    actual_icd10 TEXT NOT NULL,
    predicted_icd10 TEXT NOT NULL,
    review_status TEXT NOT NULL,
    requires_audit INTEGER NOT NULL
);

CREATE TABLE monthly_risk_metrics (
    encounter_month TEXT NOT NULL,
    risk_category TEXT NOT NULL,
    encounters INTEGER NOT NULL,
    avg_risk_score REAL NOT NULL,
    avg_confidence REAL NOT NULL,
    audit_rate REAL NOT NULL
);
