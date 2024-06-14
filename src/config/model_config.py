# ハイパーパラメータの設定
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
}

# データ分割の設定
SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42
}

# ハイパーパラメータチューニングの設定
TUNING_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
