import pytest
import pandas as pd
from src.usecase.data_preparation import (
    complement_repeated_guest_by_median,
    complement_children_by_zero,
    complement_required_car_parking_space_by_zero,
    drop_index,
    null_colum_names,
    set_one_hot_vector_of,
    calc_total_price,
    preprocess_dataset,
    preprocess_and_save_dataset,
)

def test_complement_repeated_guest_by_median():
    data = {'repeated_guest': [1, 2, None, 4, None]}
    df = pd.DataFrame(data)
    result = complement_repeated_guest_by_median(df['repeated_guest'])
    assert result.isnull().sum() == 0, "欠損値が補完されていません。"

def test_complement_children_by_zero():
    data = {'no_of_children': [1, None, 3]}
    df = pd.DataFrame(data)
    result = complement_children_by_zero(df['no_of_children'])
    assert (result == 0).sum() == 1, "欠損値が0で補完されていません。"

def test_complement_required_car_parking_space_by_zero():
    data = {'required_car_parking_space': [1, None, 3]}
    df = pd.DataFrame(data)
    result = complement_required_car_parking_space_by_zero(df['required_car_parking_space'])
    assert (result == 0).sum() == 1, "欠損値が0で補完されていません。"

def test_drop_index():
    data = {'column1': [1, None, 3], 'column2': [1, 2, None]}
    df = pd.DataFrame(data)
    result = drop_index(df)
    assert result.isnull().sum().sum() == 0, "欠損値を含む行が削除されていません。"

def test_null_colum_names():
    data = {'column1': [1, None, 3], 'column2': [1, 2, None]}
    df = pd.DataFrame(data)
    result = null_colum_names(df)
    assert set(result) == {'column1', 'column2'}, "欠損値を含む列名が正しく取得されていません。"

def test_set_one_hot_vector_of():
    data = {'type_of_meal_plan': ['A', 'B', 'A']}
    df = pd.DataFrame(data)
    result = set_one_hot_vector_of('type_of_meal_plan', df)
    assert 'type_of_meal_plan_A' in result.columns and 'type_of_meal_plan_B' in result.columns, "One-Hotエンコーディングが正しく適用されていません。"

def test_calc_total_price():
    data = {'no_of_children': [1, 2], 'no_of_adults': [2, 3], 'price_per_person': [100, 200]}
    df = pd.DataFrame(data)
    result = calc_total_price(df['no_of_children'], df['no_of_adults'], df['price_per_person'])
    assert (result == [300, 1000]).all(), "総価格が正しく計算されていません。"

def test_preprocess_dataset():
    data = {
        'repeated_guest': [1, 2, None],
        'no_of_children': [1, None, 3],
        'required_car_parking_space': [1, None, 3],
        'type_of_meal_plan': ['A', 'B', 'A'],
        'room_type_reserved': ['Single', 'Double', 'Single'],
        'no_of_adults': [2, 2, 2],
        'price_per_person': [100, 200, 150]
    }
    df = pd.DataFrame(data)
    result = preprocess_dataset(df)
    assert 'type_of_meal_plan_A' in result.columns and 'room_type_reserved_Single' in result.columns, "データセットの前処理が正しく行われていません。"
    assert 'total_price' in result.columns, "総価格が正しく計算されていません。"

def test_preprocess_and_save_dataset(mocker):
    data = {
        'repeated_guest': [1, 2, None],
        'no_of_children': [1, None, 3],
        'required_car_parking_space': [1, None, 3],
        'type_of_meal_plan': ['A', 'B', 'A'],
        'room_type_reserved': ['Single', 'Double', 'Single'],
        'no_of_adults': [2, 2, 2],
        'price_per_person': [100, 200, 150]
    }
    df = pd.DataFrame(data)
    mock_save_dataset = mocker.patch('src.usecase.data_preparation.save_dataset')
    result = preprocess_and_save_dataset(df, 'dummy_path.csv')
    mock_save_dataset.assert_called_once_with(result, 'dummy_path.csv')
    assert 'total_price' in result.columns, "総価格が正しく計算されていません。"
