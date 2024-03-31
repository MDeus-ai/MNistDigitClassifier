from data_preprocessing import data_loader
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from joblib import dump

X_train, X_test, y_train, y_test = data_loader()

# Define Model
mnist_classifier = VotingClassifier(estimators=[
    ('xgboost', XGBClassifier()),
    ('svc', SVC(probability=True)),
    ('rf', RandomForestClassifier())
], voting='soft', n_jobs=-1)

# Train The Model
mnist_classifier.fit(X_train, y_train)



# Save The Model
dump(mnist_classifier, "../models/model_v0.1.0.joblib")