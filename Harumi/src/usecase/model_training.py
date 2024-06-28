import pandas as pd

from src.interfaces.model_trainer import train_model, save_model
from src.entities.constants import PROCESSED_DATA_PATH, MODEL_PATH, TARGET_COLUMN

def train_and_save_model():
    """
    モデルの訓練を行い、訓練されたモデルを保存する。

    Returns
    -------
    None
    """
    data = pd.read_csv(PROCESSED_DATA_PATH)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    
    model, X_test, y_test = train_model(X, y)
    save_model(model, MODEL_PATH)
