import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt

def show(path):
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist.")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    plt.imshow(image)
    plt.xlabel("pixels")
    plt.ylabel("pixels")
    plt.title(path.split("/")[-1])