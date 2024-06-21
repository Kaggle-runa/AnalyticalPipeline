import pandas as pd
from interfaces.data_loader import save_dataset


def complement_repeated_guest_by_median(repeated_guest_col):

    # 対象列の中央値を計算
    fill_value = repeated_guest_col.median()
    completed_col = repeated_guest_col.fillna(fill_value)

    return completed_col


def complement_children_by_zero(no_of_children_col):

    completed_col = no_of_children_col.fillna(0)

    return completed_col


def complement_required_car_parking_space_by_zero(required_car_parking_space_col):

    completed_col = required_car_parking_space_col.fillna(0)

    return completed_col


def drop_index(df):

    return df.dropna()


def null_colum_names(df):

    count_null = df.isnull().sum()
    null_columuns = count_null[count_null > 0].index

    return null_columuns


def set_one_hot_vector_of(col_name, df):

    feature_df = pd.get_dummies(df, columns=[col_name])

    return feature_df


def calc_total_price(children_col, adults_col, price_par_person_col):

    total_proce_col = (children_col + adults_col) * price_par_person_col

    return total_proce_col


def preprocess_dataset(raw_df):

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

    preproc_df = preprocess_dataset(raw_df)
    save_dataset(preproc_df, file_path)

    return preproc_df
