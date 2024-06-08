from sklearn.metrics import accuracy_score
from joblib import load
from matplotlib import pyplot as plt

mnistClassifier = load("../models/model_v0.1.0.joblib")


"""The Prediction Function"""
def model_predict(X_test):
    return mnistClassifier.predict(X_test)


def evaluate_model(X_test, y_test):
        y_pred = mnistClassifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuarcy: {accuracy}")
