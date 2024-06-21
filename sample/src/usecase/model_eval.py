from sklearn.metrics import accuracy_score, precision_score, recall_score


def calculate_evaluation(y, pred_y):

    # 正解率の計算
    accuracy = accuracy_score(y, pred_y)
    # 適合率の計算
    precision = precision_score(y, pred_y)
    # 再現率の計算
    recall = recall_score(y, pred_y)

    evaluation_score = {"accuracy": accuracy, "precision": precision, "recall": recall}

    return evaluation_score
