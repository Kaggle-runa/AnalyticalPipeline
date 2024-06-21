import joblib

def save_model(model, file_path):
    """
    モデルを指定されたファイルパスに保存します。

    Args:
        model: 保存するモデル。
        file_path (str): 保存先のファイルパス。
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    指定されたファイルパスからモデルをロードします。

    Args:
        file_path (str): モデルが保存されているファイルパス。

    Returns:
        model: ロードされたモデル。
    """
    model = joblib.load(file_path)
    return model
