import pandas as pd

from src.entities.constants import PROCESSED_DATA_PATH
from src.interfaces.data_visualizer import plot_data


def analyze_data():
    """
    前処理済みデータの分析を行う。

    Returns
    -------
    None
    """
    data = pd.read_csv(PROCESSED_DATA_PATH)
    plot_data(data)
