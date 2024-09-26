import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_processing(file_path):
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)

    features = data.drop('median_house_value', axis=1)
    target = data['median_house_value']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
