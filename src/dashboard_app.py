import joblib
import pandas as pd
import streamlit as st
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load data and models ----
df = joblib.load('data/adult_clean.joblib')
bundle = joblib.load('models/mitigated.joblib')

baseline = bundle['baseline']
mitigator = bundle['mitigator']
scaler = bundle['scaler']

y = df['income']
X = df.drop(columns=['income'])

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)
X_scaled = scaler.transform(X_encoded)

# Sensitive features: sex and race
sensitive = df[['sex', 'race']]

# ---- Define metrics ----
metrics = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'recall': recall_score,
    'precision': precision_score,
    'selection_rate': selection_rate,
    'tpr': true_positive_rate
}

# ---- Evaluate baseline ----
yp_base = baseline.predict(X_scaled)
mf_base = MetricFrame(metrics=metrics, y_true=y, y_pred=yp_base, sensitive_features=sensitive)

# ---- Evaluate mitigator ----
yp_mitigated = mitigator.predict(X_scaled)
mf_mitigated = MetricFrame(metrics=metrics, y_true=y, y_pred=yp_mitigated, sensitive_features=sensitive)

# ---- Streamlit dashboard ----
st.title("Fairness Dashboard: Sex & Race")

st.header("Overall Metrics")
st.write("Baseline (no fairness):")
st.write(mf_base.overall)
st.write("Mitigated (fairness applied):")
st.write(mf_mitigated.overall)

st.header("Group Metrics (by Sex & Race)")
st.write("Baseline:")
st.write(mf_base.by_group)
st.write("Mitigated:")
st.write(mf_mitigated.by_group)

# ---- Bar chart: Accuracy by group ----
st.header("Comparison Plots: Accuracy by Group")
base_acc = mf_base.by_group['accuracy'].reset_index()
mitigated_acc = mf_mitigated.by_group['accuracy'].reset_index()

# Flatten multi-index for plotting
base_acc['group'] = base_acc['sex'] + " | " + base_acc['race']
mitigated_acc['group'] = mitigated_acc['sex'] + " | " + mitigated_acc['race']

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(base_acc['group'], base_acc['accuracy'], alpha=0.6, color='red', label='Baseline')
ax.bar(mitigated_acc['group'], mitigated_acc['accuracy'], alpha=0.6, color='green', label='Mitigated')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Accuracy by Sex & Race Group")
plt.legend()
st.pyplot(fig)

# ---- Heatmaps for all metrics ----
st.header("Heatmaps of Group Metrics")

# Prepare data for heatmap
def prepare_heatmap(mf):
    df_group = mf.by_group.reset_index()
    long_df = df_group.melt(id_vars=['sex', 'race'], var_name='metric', value_name='value')
    long_df['group'] = long_df['sex'] + " | " + long_df['race']
    df_pivot = long_df.pivot(index='group', columns='metric', values='value')
    return df_pivot

base_pivot = prepare_heatmap(mf_base)
mitigated_pivot = prepare_heatmap(mf_mitigated)

# Baseline heatmap
st.subheader("Baseline Model (No Fairness Mitigation)")
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(base_pivot, annot=True, cmap="Reds", fmt=".2f", ax=ax)
plt.title("Baseline Group Metrics")
st.pyplot(fig)

# Mitigated heatmap
st.subheader("Mitigated Model (Fairness Applied)")
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(mitigated_pivot, annot=True, cmap="Greens", fmt=".2f", ax=ax)
plt.title("Mitigated Group Metrics")
st.pyplot(fig)
