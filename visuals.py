import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_complexity_graph(csv_file, title=None, figsize=(14, 10), feature_extract_epochs=None,
                          loss_min=0, loss_max=2, epoch_min=None, epoch_max=90, accuracy_min=0, accuracy_max=1):
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=14)

    ax1.plot(df['loss'], label='Training Loss')
    ax1.plot(df['val_loss'], label='Validation Loss')
    ax1.set(title='Training and Validation Loss', xlabel='', ylabel='Loss')
    ax1.set_xlim([epoch_min, epoch_max])
    ax1.set_ylim([loss_min, loss_max])
    ax1.legend()

    ax2.plot(df['balanced_accuracy'], label='Training Accuracy')
    ax2.plot(df['val_balanced_accuracy'], label='Validation Accuracy')
    ax2.set(title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Balanced Accuracy')
    ax2.set_xlim([epoch_min, epoch_max])
    ax2.set_ylim([accuracy_min, accuracy_max])
    ax2.legend()

    if feature_extract_epochs is not None:
        ax1.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax2.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax1.legend()
        ax2.legend()
    
    # tight_layout() only considers ticklabels, axis labels, and titles. Thus, other artists may be clipped and also may overlap.
    # [left, bottom, right, top]
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig

def plot_grouped_2bars(scalars, scalarlabels, xticklabels, title=None, xlabel=None, ylabel=None):
    x = np.arange(len(xticklabels))  # the label locations
    width = 0.35  # the width of the bars

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    fig.patch.set_facecolor('white')
    rects1 = ax.bar(x - width/2, scalars[0], width, label=scalarlabels[0])
    rects2 = ax.bar(x + width/2, scalars[1], width, label=scalarlabels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()

def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    # References
        https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, figsize=(8, 6)):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    # References
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set(title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    im, cbar = heatmap(cm, classes, classes, ax=ax, cmap=plt.cm.Blues, cbarlabel='', grid=False)
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    return fig


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", grid=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    # References
        https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    if grid:
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
