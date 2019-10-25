import os
import numpy as np


def root():
    return f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/data"

def cache_image_indices(country):
    validate_country(country)

    path = os.path.join(root(), country, f"{country}_1000x1000_images")
    path_npy = os.path.join(root(), country, f"{country}_indices.npy")

    indices = []
    for fname in os.listdir(path):
        if fname.endswith(".tif"):
            indices.append([fname.split(".")[0].split("_")[-2:]])
                
    np.save(path_npy, indices)

def load_image_indices(country):
    validate_country(country)
    
    path = os.path.join(root(), country, f"{country}_indices.npy")
    
    if not os.path.exists(path):
        cache_image_indices(country)
        
    return np.load(path)

def validate_country(country):
    if country != "peru" and country != "kenya":
        raise ValueError("Parameter \'country\' must be one of either \'kenya\' or \'peru\'.")
