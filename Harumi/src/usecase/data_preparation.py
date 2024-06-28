from src.interfaces.data_loader import load_data
from src.interfaces.data_cleaner import clean_data
from src.entities.constants import RAW_DATA_PATH, PROCESSED_DATA_DIR, DEFAULT_PROCESSED_FILE
import os

def prepare_data():
    """
    生データを読み込み、クリーニングし、前処理済みデータとして保存する。

    Returns
    -------
    None
    """
    raw_data = load_data(RAW_DATA_PATH)
    cleaned_data = clean_data(raw_data)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, DEFAULT_PROCESSED_FILE)
    cleaned_data.to_csv(processed_file_path, index=False)
    print(f'Data processed and saved to {processed_file_path}')
