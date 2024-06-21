def pred_model(model, x_df):
    """
    モデルを使用して予測を行います。

    Args:
        model: 学習済みモデル。
        x_df (DataFrame): 特徴量データフレーム。

    Returns:
        array-like: 予測されたラベル。
    """
    pred_y = model.predict(x_df)
    return pred_y
