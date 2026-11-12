# Health Claims BI Dashboard

A business intelligence dashboard for health-insurance claims, built with **Streamlit, Python and SQL**.

**Live app:** [link]

## What it shows
- **KPIs vs. the previous period:** claims, billed amount, rejection rate and average processing days.
- **Trends:** monthly claims volume and rejection rate, plus the main rejection reasons.
- **Provider anomalies:** flags providers whose rejection rate rose more than two standard deviations above the usual change, against their own history.
- **Data quality:** validation checks on the raw feed (duplicates, missing member IDs, negative amounts, rejections without a reason) before any number reaches a report.
- **SQL:** every query behind the dashboard, visible in the app.

## Data
Synthetic data (~46,000 claims, fictional payers and providers) generated in `app.py` and loaded into an in-memory SQLite database. It includes a real-looking problem to find: one provider whose authorization rejections spike from August 2026.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Built by Sebastián López with Claude as AI coding assistant.
