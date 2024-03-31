import pandas as pd


def data_loader():
    # Import The Dataset
    train_data = pd.read_csv('../datasets/mnist_train.csv')
    test_data = pd.read_csv('../datasets/mnist_test.csv')

    # Split Dataset
    X_train, y_train = train_data.iloc[:, 1:].values, train_data.iloc[:, 0].values
    X_test, y_test = test_data.iloc[:, 1:].values, test_data.iloc[:, 0].values
    return X_train, X_test, y_train, y_test
