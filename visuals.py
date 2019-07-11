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