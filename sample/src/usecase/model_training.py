from config import model_config
from interfaces import save_model
from sklearn.tree import DecisionTreeClassifier


random_state = model_config.SEED["random_state"]
model_params = model_config.MODEL_PARAMS
dataset_config = model_config.DATASET_CONF


def split_dataset(dataset_df, test_size=0.2, random_state=100):

    split_point = int(len(dataset_df) * test_size)
    dataset_df = dataset_df.sample(frac=1, random_state=random_state)

    train_df = dataset_df.iloc[:split_point]
    test_df = dataset_df.iloc[split_point:]

    return train_df, test_df


def get_xy_dataset(data_df, target_col_name="booking_status"):

    x_df = data_df.drop([target_col_name], axis=1)
    y_df = data_df[target_col_name]

    return x_df, y_df


def train_model(train_x, train_y):

    model = DecisionTreeClassifier(**model_params)
    model.fit(train_x, train_y)

    return model, (train_x, train_y)


def train(dataset_df):

    test_size = dataset_config["test_size"]
    target_col_name = dataset_config["target_col_name"]

    train_df, test_df = split_dataset(dataset_df, test_size)

    train_x, train_y = get_xy_dataset(train_df, target_col_name=target_col_name)
    test_x, test_y = get_xy_dataset(test_df, target_col_name=target_col_name)

    model, _ = train_model(train_x, train_y)

    return model, train_x, train_y, test_x, test_y


def train_and_save_model(dataset_df, file_path):

    model, train_x, train_y, test_x, test_y = train(dataset_df)
    save_model(model, file_path)

    return model, (train_x, train_y, test_x, test_y)
