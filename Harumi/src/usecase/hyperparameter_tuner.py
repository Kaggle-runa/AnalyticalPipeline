import pandas as pd
from src.interfaces.hyperparameter_tuner import tune_hyperparameters
from src.entities.constants import PROCESSED_DATA_PATH, TARGET_COLUMN

def tune_model_hyperparameters():
    """
    ハイパーパラメータのチューニングを行う。

    Returns
    -------
    None
    """
    data = pd.read_csv(PROCESSED_DATA_PATH)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    
    best_params = tune_hyperparameters(X, y)
    print(f'Best Hyperparameters: {best_params}')