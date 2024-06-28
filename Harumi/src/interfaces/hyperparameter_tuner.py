from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.config.model_config import TUNING_PARAMS

def tune_hyperparameters(X, y):
    model = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=TUNING_PARAMS, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_params_
