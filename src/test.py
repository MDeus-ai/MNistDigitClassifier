from data_preprocessing import data_loader
from predict import model_predict, evaluate_model

X_train, X_test, y_train, y_test = data_loader()

some_digit = X_test[1976].reshape(1, -1)
print(some_digit)

model_pred = model_predict(some_digit)
print(model_pred)
evaluate_model(some_digit, y_test[1976].reshape(1, -1))