# Salifort Motors — Employee Retention Analysis

Predicting employee turnover and identifying the key drivers behind it, using HR data from a large consulting firm (Salifort Motors), to give HR a data-backed retention strategy.

## Business Problem
Salifort Motors' HR department wants to improve employee retention but doesn't know what's actually driving people to leave. This project analyzes 15,000 employee records to answer: **what factors most strongly predict whether an employee will leave, and what should HR do about it?**

## Dataset
15,000 rows, 10 features (satisfaction level, last evaluation score, number of projects, average monthly hours, tenure, work accidents, promotions in the last 5 years, department, salary tier, and whether the employee left).
Source: [Kaggle — HR Analytics and Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)

## Approach (PACE framework)
This project follows Google's PACE methodology — **Plan, Analyze, Construct, Execute** — a structured approach to going from raw business question to stakeholder-ready recommendations.

**Plan:** Framed the business question and identified stakeholders (HR leadership) before touching the data.

**Analyze:** Cleaned the data (removed ~3,000 duplicate rows, standardized column names, checked for outliers using IQR) and ran exploratory analysis — distribution plots, boxplots, a violin plot comparing satisfaction across salary tiers, a correlation clustermap, and a PairGrid across the key numeric features — to understand relationships before modeling.

**Construct:** Built two models to predict turnover:
- **Logistic Regression** (balanced class weights, since ~83% of employees stayed vs. ~17% who left — an imbalanced target)
- **Random Forest**, tuned via `GridSearchCV` across max depth, min samples split/leaf, max features, and number of estimators

**Execute:** Evaluated both models on accuracy, precision, recall, F1, and AUC (chosen over plain accuracy because of the class imbalance), then translated the winning model's feature importances into concrete HR recommendations.

## Key Findings
- **Satisfaction level** was the single strongest predictor of turnover — clear negative correlation, lower satisfaction meant far higher turnover risk.
- **Workload indicators** (number of projects and average monthly hours) were the next strongest drivers — overworked employees left at meaningfully higher rates.
- Random Forest outperformed Logistic Regression on the evaluation metrics used.

## Recommendations
- Cap the number of concurrent projects per employee.
- Investigate why 4-year-tenured employees show elevated dissatisfaction, and consider promotion pathways at that stage.
- Reward long working hours proportionately, or stop expecting them by default — don't let high evaluation scores implicitly require 200+ monthly hours.
- Clarify overtime pay policy and workload expectations company-wide.

## Tech Stack
Python, pandas, NumPy, scikit-learn (Logistic Regression, Random Forest, GridSearchCV), Matplotlib, Seaborn

## Repo Contents
- `notebook/salifort_motors_analysis.ipynb` — full analysis, from data cleaning through model evaluation
- `images/` — key visualizations referenced above

## What I'd Explore Next
Feature engineering around tenure × workload interaction effects, and testing an XGBoost model against the current Random Forest baseline.

---
*This project was completed as the capstone for the Google Advanced Data Analytics Professional Certificate.*
