# app/interfaces/data_visualizer.py
import matplotlib.pyplot as plt

def plot_data(df):
    df.plot()
    plt.show()
