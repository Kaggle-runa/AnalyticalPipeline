from sklearn import tree

from matplotlib import pyplot as plt
import numpy as np


def plot_feature_importances(model, x):
    # 特徴量の重要度の棒グラフ

    feature = model.feature_importances_
    label = x.columns
    indices = np.argsort(feature)

    _ = plt.figure(figsize=(10, 10))

    plt.barh(range(len(feature)), feature[indices])

    plt.yticks(range(len(feature)), label[indices], fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Feature", fontsize=18)
    plt.xlabel("Feature Importance", fontsize=18)


def visualize_tree(model, feature_colmuns):

    tree.plot_tree(model, feature_names=feature_colmuns, filled=True)
    plt.show()
