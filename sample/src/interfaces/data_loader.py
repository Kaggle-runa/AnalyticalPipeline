import pandas as pd

def raw_data_loader(path = "./data/raw/dataset.csv"):
    """
    生データをCSVファイルから読み込み、DataFrameとして返します。

    Args:
        path (str): CSVファイルのパス。デフォルトは"./data/raw/dataset.csv"。

    Returns:
        DataFrame: 読み込んだデータ。
    """
    raw_df = pd.read_csv(path)
    return raw_df

def save_dataset(df, file_path):
    """
    データセットをCSVファイルに保存します。

    Args:
        df (DataFrame): 保存するデータ。
        file_path (str): 保存先のファイルパス。
    """
    df.to_csv(file_path)
