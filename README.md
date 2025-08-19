# Bias & Fairness in ML — Adult Income

**Goal:** Train classifiers, measure bias across sensitive attributes, mitigate unfairness, and visualize accuracy–fairness tradeoffs in a Streamlit dashboard.

## Quickstart
```bash
pip install -r requirements.txt
# Prepare data (downloads Adult dataset via sklearn)
python src/data_prep.py
# Train baseline models
python src/train.py
# Compute fairness metrics + generate report
python src/fairness_metrics.py
# Mitigate bias and compare
python src/mitigate.py
# Launch dashboard
streamlit run src/dashboard_app.py
# Or run all
python src/pipeline.py
```
