import cv2
import numpy as np
import os, sys

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from modules.data import util
                
def downscale(country, D, subsampling=0, quality=90):

    i_path = os.path.join(util.root(), country, f"{country}_1000x1000_images")
    o_path = os.path.join(util.root(), country, f"{D}_downscale_images")
    
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    
    errors = []
    
    for i, fname in enumerate(os.listdir(i_path)):
        if i % 1000 == 0:
            print(f"Processed {i} file dscriptors.")
        if os.path.isfile(os.path.join(i_path, fname)):
            try:
                im = Image.open(os.path.join(i_path, fname))
            except:
                errors.append(fname)
                continue

            head, _ = os.path.splitext(fname)
            head = "_".join(head.split("_")[-2:])
            
            im = im.resize((D, D)).convert("RGB")

            im.save(os.path.join(o_path, f"{head}.jpg"), "JPEG", subsampling=subsampling, quality=quality)
            
    with open(os.path.join(o_path, "errors.txt"), "w") as o_err:
        for e in errors:
            o_err.write(f"{e}\n")


def downcrop(country, D, subsampling=0, quality=90):

    i_path = os.path.join(util.root(), country, f"{country}_1000x1000_images")
    o_path = os.path.join(util.root(), country, f"{D}_downcrop_images")
    
    if not os.path.exists(o_path):
        os.makedirs(o_path)
        
    greater = 1000 // 2 + D // 2
    smaller = 1000 // 2 - D // 2
    
    errors = []
    
    for i, fname in enumerate(os.listdir(i_path)):
        if i % 1000 == 0:
            print(f"Processed {i} file dscriptors.")
        if os.path.isfile(os.path.join(i_path, fname)):
            try:
                im = Image.open(os.path.join(i_path, fname))
            except:
                errors.append(fname)
                continue

            head, _ = os.path.splitext(fname)
            head = "_".join(head.split("_")[-2:])
                        
            im = im.crop((smaller, smaller, greater, greater)).convert("RGB")
        
            im.save(os.path.join(o_path, f"{head}.jpg"), "JPEG", subsampling=subsampling, quality=quality)
            
            
    with open(os.path.join(o_path, "errors.txt"), "w") as o_err:
        for e in errors:
            o_err.write(f"{e}\n")