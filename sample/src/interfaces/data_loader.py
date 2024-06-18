import pandas as pd

def raw_data_loader(path = "./data/raw/dataset.csv"):

    raw_df = pd.read_csv(path)

    return raw_df

def save_dataset(df, file_path):

    df.to_csv(file_path)