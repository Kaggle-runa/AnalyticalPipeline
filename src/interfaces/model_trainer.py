import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.config.model_config import MODEL_PARAMS, SPLIT_PARAMS

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_PARAMS['test_size'], random_state=SPLIT_PARAMS['random_state'])
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, file_path):
    joblib.dump(model, file_path)