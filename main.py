from scripts.preprocessing import data_processing
from scripts.linear_regression import linear_regression, predict, r2_score

file_path = 'data/california_housing_train.csv'
X_train, X_test, y_train, y_test = data_processing(file_path)

theta = linear_regression(X_train, y_train)

y_pred = predict(X_test, theta)

r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2:.4f}')
