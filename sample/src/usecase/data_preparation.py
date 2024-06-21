import pandas as pd
from src.interfaces.data_loader import save_dataset

def complement_repeated_guest_by_median(repeated_guest_col):
    """
    repeated_guest列の欠損値を中央値で補完します。

    Args:
        repeated_guest_col (Series): 対象列。

    Returns:
        Series: 補完された列。
    """
    fill_value = repeated_guest_col.median()
    completed_col = repeated_guest_col.fillna(fill_value)
    return completed_col

def complement_children_by_zero(no_of_children_col):
    """
    no_of_children列の欠損値を0で補完します。

    Args:
        no_of_children_col (Series): 対象列。

    Returns:
        Series: 補完された列。
    """
    completed_col = no_of_children_col.fillna(0)
    return completed_col

def complement_required_car_parking_space_by_zero(required_car_parking_space_col):
    """
    required_car_parking_space列の欠損値を0で補完します。

    Args:
        required_car_parking_space_col (Series): 対象列。

    Returns:
        Series: 補完された列。
    """
    completed_col = required_car_parking_space_col.fillna(0)
    return completed_col

def drop_index(df):
    """
    DataFrameから欠損値を持つ行を削除します。

    Args:
        df (DataFrame): 対象データフレーム。

    Returns:
        DataFrame: 欠損値を持つ行が削除されたデータフレーム。
    """
    return df.dropna()

def null_colum_names(df):
    """
    DataFrameの欠損値を持つ列名を取得します。

    Args:
        df (DataFrame): 対象データフレーム。

    Returns:
        Index: 欠損値を持つ列名のリスト。
    """
    count_null = df.isnull().sum()
    null_columuns = count_null[count_null > 0].index
    return null_columuns

def set_one_hot_vector_of(col_name, df):
    """
    指定した列に対してOne-Hotエンコーディングを適用します。

    Args:
        col_name (str): 対象列名。
        df (DataFrame): 対象データフレーム。

    Returns:
        DataFrame: One-Hotエンコーディングが適用されたデータフレーム。
    """
    feature_df = pd.get_dummies(df, columns=[col_name])
    return feature_df

def calc_total_price(children_col, adults_col, price_par_person_col):
    """
    子供と大人の人数、および一人当たりの価格から総価格を計算します。

    Args:
        children_col (Series): 子供の人数列。
        adults_col (Series): 大人の人数列。
        price_par_person_col (Series): 一人当たりの価格列。

    Returns:
        Series: 総価格列。
    """
    total_proce_col = (children_col + adults_col) * price_par_person_col
    return total_proce_col

def preprocess_dataset(raw_df):
    """
    データセットを前処理します。

    Args:
        raw_df (DataFrame): 生データフレーム。

    Returns:
        DataFrame: 前処理されたデータフレーム。
    """
    raw_df["repeated_guest"] = complement_repeated_guest_by_median(
        raw_df["repeated_guest"]
    )
    raw_df["no_of_children"] = complement_children_by_zero(raw_df["no_of_children"])
    raw_df["required_car_parking_space"] = (
        complement_required_car_parking_space_by_zero(
            raw_df["required_car_parking_space"]
        )
    )

    completed_df = drop_index(raw_df)

    preproc_df = set_one_hot_vector_of("type_of_meal_plan", completed_df)
    preproc_df = set_one_hot_vector_of("room_type_reserved", preproc_df)

    preproc_df["total_price"] = calc_total_price(
        completed_df["no_of_children"],
        completed_df["no_of_adults"],
        completed_df["price_per_person"],
    )

    return preproc_df

def preprocess_and_save_dataset(raw_df, file_path):
    """
    データセットを前処理し、CSVファイルとして保存します。

    Args:
        raw_df (DataFrame): 生データフレーム。
        file_path (str): 保存先のファイルパス。

    Returns:
        DataFrame: 前処理されたデータフレーム。
    """
    preproc_df = preprocess_dataset(raw_df)
    save_dataset(preproc_df, file_path)
    return preproc_df
