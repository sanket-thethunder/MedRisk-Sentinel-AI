# MedRisk Sentinel AI

End-to-end healthcare analytics project that simulates a Cotiviti-style Risk Adjustment workflow using NLP, SQL, Python, and Streamlit.

## What This Project Does

MedRisk Sentinel AI classifies synthetic clinical-style text into HCC-style risk categories, simulates ICD-10 assignment, reconciles false positive and false negative coding outcomes, and visualizes operational quality metrics in an interactive dashboard.

The project is designed around a healthcare data analytics internship role focused on:

- Risk Adjustment analytics
- Precision and recall calculations
- False positive and false negative ICD-10 code review
- HCC-style metrics
- SQL analysis over healthcare data
- Python automation
- NLP and AI-assisted medical data processing
- BI dashboarding for operational KPIs

## Business Problem

Healthcare risk adjustment teams need to validate whether medical conditions are accurately captured from claims and clinical documentation. Missed ICD-10 codes can reduce recall and understate member risk. Incorrectly assigned codes can reduce precision and create audit exposure.

This project answers:

- Which ICD-10 categories have the weakest precision or recall?
- Which clinical records should be prioritized for audit review?
- How do HCC-style risk categories trend over time?
- Can NLP classify unstructured clinical text into risk categories?
- How can analysts monitor model quality and coding reconciliation in one workflow?

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- TF-IDF NLP pipeline
- Logistic Regression classifier
- SQLite SQL analytics layer
- Streamlit dashboard
- Plotly visualizations

## Project Structure

```text
MedRisk-Sentinel-AI/
  app/
    streamlit_app.py
  data/
    synthetic_encounters.csv
    code_reconciliation.csv
    monthly_risk_metrics.csv
    medrisk_sentinel.db
  models/
    risk_category_tfidf_logreg.joblib
  reports/
    classification_report.txt
    icd10_precision_recall_metrics.csv
    model_metrics.json
    monthly_audit_queue_metrics.csv
    nlp_scored_holdout.csv
  sql/
    analytics_queries.sql
    schema.sql
  src/
    data_generator.py
    model_pipeline.py
    sql_metrics.py
  tests/
    test_metrics.py
```

## How To Run

```bash
pip install -r requirements.txt
python src/data_generator.py
python src/model_pipeline.py
python src/sql_metrics.py
streamlit run app/streamlit_app.py
```

## Dashboard Features

- Encounter volume, precision, recall, F1, accuracy, and audit queue KPIs
- HCC-style risk category distribution
- Average risk score trend by month
- False positive and false negative reconciliation views
- ICD-10 level precision and recall table
- Audit queue table for low-confidence or incorrect code assignments
- Clinical text review table for explainable analyst workflows

## Resume Bullets

- Built an end-to-end healthcare Risk Adjustment analytics platform using Python, SQL, NLP, scikit-learn, and Streamlit to classify clinical-style text into HCC-style risk categories and simulate ICD-10 assignment workflows.
- Designed SQL-backed reconciliation metrics to calculate precision, recall, F1, false positive, and false negative rates across ICD-10 codes, supporting audit prioritization and coding quality analysis.
- Developed an interactive Streamlit dashboard for operational KPIs, model performance trends, HCC-style category distributions, and low-confidence encounter review, mirroring healthcare data analytics and AI/NLP use cases.

## Interview Talking Points

- Explain why precision matters for avoiding incorrect codes and why recall matters for reducing missed conditions.
- Show how SQL tables separate encounters, code reconciliation, and monthly risk metrics.
- Walk through the NLP pipeline: TF-IDF feature extraction plus Logistic Regression classification.
- Discuss how the audit queue combines model confidence and reconciliation status.
- Connect the project to enterprise healthcare analytics: claims, clinical text, ICD-10, HCC-style categories, QA, and BI reporting.

## Data Note

All data is synthetic and generated locally for portfolio use. It does not contain real patient information or PHI.
