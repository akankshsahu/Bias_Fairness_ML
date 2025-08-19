# Bias Mitigation Dashboard

This project demonstrates bias mitigation in machine learning models using the Fairlearn library. I trained a baseline logistic regression model on a locally created Adult Income dataset and then applied Demographic Parity constraints using the Exponentiated Gradient mitigator to reduce bias across sensitive groups. I plan to pull genuine labor values soon.

The Streamlit dashboard displays:

  -Overall metrics: accuracy, F1 score, recall, precision, selection rate, and true positive rate.

  -Group-level metrics: the same metrics broken down by combinations of sensitive attributes (sex and race).

  -Visualizations: bar charts and heatmaps to compare baseline vs. mitigated model performance across groups, highlighting the effect of fairness mitigation.

This setup allows easy exploration of fairness trade-offs in predictive modeling while maintaining transparency in performance across demographic groups

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
