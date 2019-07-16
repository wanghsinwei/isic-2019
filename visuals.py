import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_complexity_graph(csv_file, figsize=(20, 5)):
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor('white')

    ax1.plot(df['loss'], label='Training loss')
    ax1.plot(df['val_loss'], label='Validation loss')
    ax1.set(xlabel='epoch', ylabel='Loss')
    ax1.legend()

    ax2.plot(df['balanced_accuracy'], label='Training Accuracy')
    ax2.plot(df['val_balanced_accuracy'], label='Validation Accuracy')
    ax2.set(xlabel='epoch', ylabel='Balanced Accuracy')
    ax2.legend()

def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    Ref: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')