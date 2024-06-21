import pytest
import pandas as pd
from src.interfaces.data_loader import raw_data_loader, save_dataset

def test_raw_data_loader(mocker):
    # モックデータ
    data = {'column1': [1, 2], 'column2': [3, 4]}
    mock_df = pd.DataFrame(data)

    # pd.read_csvをモックしてモックデータを返すようにする
    mocker.patch('pandas.read_csv', return_value=mock_df)

    # 関数を呼び出して結果を確認
    result = raw_data_loader('dummy_path.csv')
    assert result.equals(mock_df), "DataFrameが正しく読み込まれていません。"

def test_save_dataset(mocker):
    # モックデータ
    data = {'column1': [1, 2], 'column2': [3, 4]}
    mock_df = pd.DataFrame(data)

    # df.to_csvをモックする
    mock_to_csv = mocker.patch('pandas.DataFrame.to_csv')

    # 関数を呼び出して保存を確認
    save_dataset(mock_df, 'dummy_path.csv')
    mock_to_csv.assert_called_once_with('dummy_path.csv')
