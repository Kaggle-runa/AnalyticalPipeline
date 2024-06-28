import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.usecase.model_eval import calculate_evaluation

def test_calculate_evaluation():
    y = [0, 1, 1, 0, 1]
    pred_y = [0, 1, 0, 0, 1]

    result = calculate_evaluation(y, pred_y)

    expected_accuracy = accuracy_score(y, pred_y)
    expected_precision = precision_score(y, pred_y)
    expected_recall = recall_score(y, pred_y)

    assert result["accuracy"] == expected_accuracy, "正解率が正しく計算されていません。"
    assert result["precision"] == expected_precision, "適合率が正しく計算されていません。"
    assert result["recall"] == expected_recall, "再現率が正しく計算されていません。"
