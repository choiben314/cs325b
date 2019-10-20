import os
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

def show(path):
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist.")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    plt.imshow(image)
    plt.xlabel("pixels")
    plt.ylabel("pixels")
    plt.title(path.split("/")[-1])
    
def plot_channel_hist(image, visualize=True):
    channels = cv2.split(image)
    colors = ["r", "g", "b"]
    histograms = []
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        histograms.append(hist)
    
    if visualize:
        plt.plot(histograms[0], color="r")
        plt.plot(histograms[1], color="g")
        plt.plot(histograms[2], color="b")
        plt.xlabel("RGB Value")
        plt.ylabel("Pixel Density")
        plt.title("Color Histogram")
        
    return histograms

def plot_confusion_matrix(y_true, y_pred, classes, name, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [classes[i] for i in unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.show()
    plt.ylim([2.5, -0.5])
    plt.title(name)
    
    return cm