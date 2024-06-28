import pytest
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from src.usecase.data_visualizer import plot_feature_importances, visualize_tree
from matplotlib import pyplot as plt

def test_plot_feature_importances(mocker):
    # モックデータ
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    df = pd.DataFrame(data)
    model = DecisionTreeClassifier()
    model.fit(df, [0, 1, 0])

    # プロット関数をモックする
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.barh')
    mocker.patch('matplotlib.pyplot.yticks')
    mocker.patch('matplotlib.pyplot.xticks')
    mocker.patch('matplotlib.pyplot.ylabel')
    mocker.patch('matplotlib.pyplot.xlabel')

    plot_feature_importances(model, df)

    # 関数が正しく呼び出されたことを確認
    plt.barh.assert_called()
    plt.yticks.assert_called()
    plt.xticks.assert_called()
    plt.ylabel.assert_called()
    plt.xlabel.assert_called()

def test_visualize_tree(mocker):
    # モックデータ
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    df = pd.DataFrame(data)
    model = DecisionTreeClassifier()
    model.fit(df, [0, 1, 0])

    # プロット関数をモックする
    mocker.patch('sklearn.tree.plot_tree')
    mocker.patch('matplotlib.pyplot.show')

    visualize_tree(model, df.columns)

    # 関数が正しく呼び出されたことを確認
    tree.plot_tree.assert_called_with(model, feature_names=df.columns, filled=True)
    plt.show.assert_called()
