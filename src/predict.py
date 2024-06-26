from sklearn.metrics import accuracy_score
from joblib import load
import numpy as np

mnistClassifier = load("../models/model_v0.1.0.joblib")


"""The Prediction Function"""
def model_predict(X_test):
    return mnistClassifier.predict(X_test), np.max(mnistClassifier.predict_proba(X_test), axis=1)
