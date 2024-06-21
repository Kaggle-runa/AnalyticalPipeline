import pytest
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from src.usecase.model_predict import pred_model

def test_pred_model():
    # モックデータ
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    df = pd.DataFrame(data)
    model = DecisionTreeClassifier()
    model.fit(df, [0, 1, 0])

    pred_y = pred_model(model, df)

    expected_pred_y = model.predict(df)
    assert (pred_y == expected_pred_y).all(), "モデルの予測が正しく行われていません。"
