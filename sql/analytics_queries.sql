-- Code-level precision, recall, and F1 for ICD-10 reconciliation.
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
    ROUND(1.0 * true_positive / NULLIF(true_positive + false_negative, 0), 3) AS recall
FROM code_metrics
ORDER BY recall ASC;

-- Monthly operational QA view for audit prioritization.
SELECT
    encounter_month,
    risk_category,
    COUNT(*) AS encounters,
    SUM(CASE WHEN review_status = 'False Positive' THEN 1 ELSE 0 END) AS false_positive_count,
    SUM(CASE WHEN review_status = 'False Negative' THEN 1 ELSE 0 END) AS false_negative_count,
    ROUND(AVG(risk_score), 3) AS avg_risk_score,
    ROUND(AVG(model_confidence), 3) AS avg_model_confidence
FROM encounters
GROUP BY encounter_month, risk_category
ORDER BY encounter_month, false_negative_count DESC;
