import pandas as pd, numpy as np, os, joblib
from sklearn.datasets import fetch_openml
import ssl
import certifi

def custom_https_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = custom_https_context

os.makedirs('data', exist_ok=True)

def load_adult():
    adult = fetch_openml('adult', version=2, as_frame=True)
    df = adult.frame
    # Clean target
    df['income'] = df['class'].map({'>50K':1,'<=50K':0})
    df = df.drop(columns=['class'])
    # Keep a subset of columns
    cols = ['age','workclass','education','marital-status','occupation','relationship','race','sex','hours-per-week','native-country','income']
    df = df[cols].dropna()
    return df

df = load_adult()
joblib.dump(df, 'data/adult_clean.joblib')
print('Saved data/adult_clean.joblib with shape', df.shape)
