import os
import pandas as pd

# csvファイルを読み込む関数
def load_data(file_path):
    return pd.read_csv(file_path)

def list_processed_files(processed_data_dir :str):
    """
    指定されたディレクトリ内に存在するファイルのリストを取得する。
    
    Parameters
    ----------
    processed_data_dir : str
        前処理済みデータが格納されているディレクトリのパス。
    
    Returns
    -------
    list
        指定されたディレクトリ内に存在するファイル名のリスト。
    """
    return [f for f in os.listdir(processed_data_dir) if os.path.isfile(os.path.join(processed_data_dir, f))]
