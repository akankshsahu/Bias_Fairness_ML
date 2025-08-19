import joblib, pandas as pd, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


df = joblib.load('data/adult_clean.joblib')
y = df['income']
X = df.drop(columns=['income'])

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

pre = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

clf = Pipeline([
    ('pre', pre),
    ('lr', LogisticRegression(max_iter=1000))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(Xtr, ytr)
os.makedirs('models', exist_ok=True)
joblib.dump({'model':clf,'num_cols':num_cols,'cat_cols':cat_cols}, 'models/baseline.joblib')

pred = clf.predict(Xte)
print('Accuracy:', accuracy_score(yte, pred), 'F1:', f1_score(yte, pred))
