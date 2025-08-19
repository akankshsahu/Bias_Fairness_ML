import joblib
import pandas as pd
import numpy as np
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Suppress FutureWarnings from Fairlearn (chained assignment)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load baseline model and data
bundle = joblib.load('models/baseline.joblib')
model = bundle['model']  # keep this if you want reference to original baseline

df = joblib.load('data/adult_clean.joblib')
y = df['income']
X = df.drop(columns=['income'])

# Split train/test
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode categorical features
Xtr_encoded = pd.get_dummies(Xtr, drop_first=True)
Xte_encoded = pd.get_dummies(Xte, drop_first=True)

# Align columns in case some categories are missing in test
Xte_encoded = Xte_encoded.reindex(columns=Xtr_encoded.columns, fill_value=0)

# Scale features
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_encoded)
Xte_scaled = scaler.transform(Xte_encoded)

# ---- Baseline (no fairness constraints) ----
baseline = LogisticRegression(max_iter=5000)
baseline.fit(Xtr_scaled, ytr)
yp_base = baseline.predict(Xte_scaled)
print('Baseline Accuracy:', accuracy_score(yte, yp_base))
print('Baseline F1:', f1_score(yte, yp_base))

# ---- Mitigator (fairness constraints) ----
constraint = DemographicParity()

# Include both sex and race as sensitive features
sensitive_tr = Xtr[['sex', 'race']]

mitigator = ExponentiatedGradient(LogisticRegression(max_iter=5000), constraint)
mitigator.fit(Xtr_scaled, ytr, sensitive_features=sensitive_tr)
yp_mitigated = mitigator.predict(Xte_scaled)
print('Mitigated Accuracy:', accuracy_score(yte, yp_mitigated))
print('Mitigated F1:', f1_score(yte, yp_mitigated))

# ---- Save both models ----
joblib.dump({
    'baseline': baseline,
    'mitigator': mitigator,
    'scaler': scaler
}, 'models/mitigated.joblib')
