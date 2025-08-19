import streamlit as st
from sklearn.metrics import accuracy_score, recall_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio
)


def compute_and_display_fairness(y_true, y_pred, sensitive_features, group_name="Group"):
    # --- Group + Overall metrics ---
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "tpr": recall_score,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    # --- Fairness gaps/ratios ---
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = mf.by_group["tpr"].max() - mf.by_group["tpr"].min()

    # --- Streamlit Display ---
    st.subheader("Overall Metrics")
    st.write(mf.overall)

    st.subheader(f"Metrics by {group_name}")
    st.write(mf.by_group)

    st.subheader("Fairness Gaps / Ratios")
    st.write("**Demographic Parity Difference:**", round(dp_diff, 4))
    st.write("**Demographic Parity Ratio:**", round(dp_ratio, 4))
    st.write("**Equal Opportunity Difference:**", round(eo_diff, 4))

    return mf, dp_diff, dp_ratio, eo_diff
