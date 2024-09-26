import numpy as np


def linear_regression(X_train, y_train):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    return theta


def predict(X_test, theta):
    X_b_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    return X_b_test.dot(theta)


def r2_score(y_true, y_pred):
    total_variance = sum((y_true - y_true.mean()) ** 2)
    residual_variance = sum((y_true - y_pred) ** 2)
    return 1 - residual_variance / total_variance
