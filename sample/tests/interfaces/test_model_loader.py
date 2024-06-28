import pytest
import joblib
from src.interfaces.model_loader import save_model, load_model

def test_save_model(mocker):
    # モックモデル
    model = 'dummy_model'

    # joblib.dumpをモックする
    mock_dump = mocker.patch('joblib.dump')

    # 関数を呼び出して保存を確認
    save_model(model, 'dummy_model_path.pkl')
    mock_dump.assert_called_once_with(model, 'dummy_model_path.pkl')

def test_load_model(mocker):
    # モックモデル
    model = 'dummy_model'

    # joblib.loadをモックしてモックモデルを返すようにする
    mocker.patch('joblib.load', return_value=model)

    # 関数を呼び出して結果を確認
    result = load_model('dummy_model_path.pkl')
    assert result == model, "モデルが正しくロードされていません。"
