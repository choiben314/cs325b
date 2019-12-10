import os
import sys
import cv2
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage import draw
from skimage.transform import resize
from scipy.spatial import cKDTree

from modules.data import util
from modules.data import DataManager

def downscale(country, D, subsampling=0, quality=90):

    i_path = os.path.join(util.root(), country, f"{country}_1000x1000_images")
    o_path = os.path.join(util.root(), country, f"{D}", "scaled")

    if not os.path.exists(o_path):
        os.makedirs(o_path)

    errors = []

    for i, fname in enumerate(os.listdir(i_path)):
        if i % 1000 == 0:
            print(f"Processed {i} file descriptors.")
        if os.path.isfile(os.path.join(i_path, fname)):
            try:
                im = Image.open(os.path.join(i_path, fname))
            except:
                errors.append(fname)
                continue

            head, _ = os.path.splitext(fname)
            head = head.split("_")[-1]

            im = im.resize((D, D)).convert("RGB")

            im.save(os.path.join(o_path, f"{head}.jpg"), "JPEG", subsampling=subsampling, quality=quality)

    with open(os.path.join(o_path, "errors.txt"), "w") as o_err:
        for e in errors:
            o_err.write(f"{e}\n")

def downcrop(country, D, subsampling=0, quality=90):

    i_path = os.path.join(util.root(), country, f"{country}_1000x1000_images")
    o_path = os.path.join(util.root(), country, f"{D}", "cropped")

    if not os.path.exists(o_path):
        os.makedirs(o_path)

    greater = 1000 // 2 + D // 2
    smaller = 1000 // 2 - D // 2

    errors = []

    for i, fname in enumerate(os.listdir(i_path)):
        if i % 1000 == 0:
            print(f"Processed {i} file descriptors.")
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

def polygon_perimeter(r, c):
    r = np.round(r).astype(int)
    c = np.round(c).astype(int)

    rr, cc = [], []
    for i in range(len(r) - 1):
        line_r, line_c = draw.line(r[i], c[i], r[i + 1], c[i + 1])
        rr.extend(line_r)
        cc.extend(line_c)

    rr = np.asarray(rr)
    cc = np.asarray(cc)

    return rr, cc

def generate_masks(country, config, data_manager=None, threshold=20):

    if data_manager is None:
        data_manager = DataManager(config)

    orig_dim = 1000
    crop_dim = config["image_size"]

    o_path = os.path.join(util.root(), country, str(crop_dim), "masks")

    if not os.path.exists(o_path):
        os.makedirs(o_path)

    points = [(i, j) for i in range(orig_dim) for j in range(orig_dim)]
    point_tree = cKDTree(points)

    for idx in data_manager.dataframes[country].index.values:

        min_lat = float(data_manager.dataframes[country].loc[idx]["minlat"])
        max_lat = float(data_manager.dataframes[country].loc[idx]['maxlat'])
        min_lon = float(data_manager.dataframes[country].loc[idx]['minlon'])
        max_lon = float(data_manager.dataframes[country].loc[idx]['maxlon'])

        id1 = idx
        id2 = int(data_manager.dataframes[country]["id"][id1])

        points_inline = []
        for lon, lat in data_manager.shapefiles[country].shape(idx).points:
            if min_lat < lat < max_lat and min_lon < lon < max_lon:
                points_inline.append((lon, lat))
        if len(points_inline) == 0:
            continue

        lon, lat = zip(*points_inline)
        lat = orig_dim * (np.array(lat) - min_lat) / (max_lat - min_lat)
        lon = orig_dim * (np.array(lon) - min_lon) / (max_lon - min_lon)

        lon_pixel, lat_pixel = polygon_perimeter(lon, lat)

        coords = []
        lon_thick = []
        lat_thick = []

        for i in range(len(lon_pixel)):
            nearest_points_idx = point_tree.query_ball_point([lon_pixel[i], lat_pixel[i]], threshold)
            for point_idx in nearest_points_idx:
                lon, lat = points[point_idx]
                coords.append((lon, lat))
                lon_thick.append(lon)
                lat_thick.append(lat)

        img = np.zeros((orig_dim, orig_dim), dtype=np.uint8)
        img[lat_thick, lon_thick] = 1
        save_img = img[(orig_dim // 2 - crop_dim // 2):(orig_dim // 2 + crop_dim // 2), (orig_dim // 2 - crop_dim // 2):(orig_dim // 2 + crop_dim // 2)]

        cv2.imwrite(os.path.join(o_path, f"{id1}_{id2}.png"), save_img)

def generate_filters(country, value_threshold=150):

    i_path = os.path.join(util.root(), country, f"{country}_1000x1000_images")

    corrupt = open(os.path.join(util.root(), country, "corrupt.txt"), "w")
    clouded = open(os.path.join(util.root(), country, "cloudy.txt"), "w")
    grayscale = open(os.path.join(util.root(), country, "grayscale.txt"), "w")

    for i, fname in enumerate(os.listdir(i_path)):
        img = cv2.imread(os.path.join(i_path, fname), cv2.IMREAD_COLOR)
        fname += "\n"
        if img is None:
            corrupt.write(fname)
        else:
            means = list(cv2.mean(img))[:-1]
            if all(value > value_threshold for value in means):
                clouded.write(fname)
            elif means.count(means[0]) == len(means):
                grayscale.write(fname)

    corrupt.close()
    clouded.close()
    grayscale.close()
