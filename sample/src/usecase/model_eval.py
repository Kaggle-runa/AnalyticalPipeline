from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_evaluation(y, pred_y):
    """
    モデルの評価指標を計算します。

    Args:
        y (array-like): 真のラベル。
        pred_y (array-like): 予測されたラベル。

    Returns:
        dict: 正解率、適合率、再現率を含む評価指標の辞書。
    """
    accuracy = accuracy_score(y, pred_y)
    precision = precision_score(y, pred_y)
    recall = recall_score(y, pred_y)

    evaluation_score = {"accuracy": accuracy, "precision": precision, "recall": recall}

    return evaluation_score

