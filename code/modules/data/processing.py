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

            head = os.path.splitext(fname)[0].split("_")[-1]

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
        if os.path.isfile(os.path.join(i_path, fname)):
            try:
                im = Image.open(os.path.join(i_path, fname))
            except:
                errors.append(fname)
                continue

            head = os.path.splitext(fname)[0].split("_")[-1]

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

def generate_masks(country, config, data_manager, threshold=20):

    orig_dim = 1000
    crop_dim = config["image_size"]

    o_path = os.path.join(util.root(), country, str(crop_dim), "masks")

    if not os.path.exists(o_path):
        os.makedirs(o_path)

    points = [(i, j) for i in range(orig_dim) for j in range(orig_dim)]
    point_tree = cKDTree(points)

    images = []

    for i in range(data_manager.dataframes[country].index.values.shape[0]):

        ord = int(data_manager.dataframes[country]["ord"].iloc[i])
        id1 = data_manager.dataframes[country].index.values[i]
        id2 = int(data_manager.dataframes[country]["id"].iloc[i])

        min_lat = float(data_manager.dataframes[country].iloc[i]["minlat"])
        max_lat = float(data_manager.dataframes[country].iloc[i]['maxlat'])
        min_lon = float(data_manager.dataframes[country].iloc[i]['minlon'])
        max_lon = float(data_manager.dataframes[country].iloc[i]['maxlon'])

        points_inline = []
        for lon, lat in data_manager.shapefiles[country].shape(id1).points:
            if min_lat < lat < max_lat and min_lon < lon < max_lon:
                points_inline.append((lon, lat))

        if len(points_inline) > 0:
            lon, lat = zip(*points_inline)
        else:
            lon, lat = [], []
        lat = orig_dim * (np.array(lat) - min_lat) / (max_lat - min_lat)
        lon = orig_dim * (np.array(lon) - min_lon) / (max_lon - min_lon)

        lon_pixel, lat_pixel = polygon_perimeter(lon, lat)

        coords = []
        lon_thick = []
        lat_thick = []

        for p_i in range(len(lon_pixel)):
            nearest_points_idx = point_tree.query_ball_point([lon_pixel[p_i], lat_pixel[p_i]], threshold)
            for point_idx in nearest_points_idx:
                lon, lat = points[point_idx]
                coords.append((lon, lat))
                lon_thick.append(lon)
                lat_thick.append(lat)

        img = np.zeros((orig_dim, orig_dim), dtype=np.uint8)
        img[lat_thick, lon_thick] = 1
        save_img = img[
            (orig_dim // 2 - crop_dim // 2):(orig_dim // 2 + crop_dim // 2),
            (orig_dim // 2 - crop_dim // 2):(orig_dim // 2 + crop_dim // 2)
        ]

        if country == "kenya":
            cv2.imwrite(os.path.join(o_path, f"{id2}.png"), save_img)
        elif country == "peru":
            ord = int(data_manager.dataframes[country]["ord"].iloc[i])
            cv2.imwrite(os.path.join(o_path, f"{id2}-{ord}.png"), save_img)

        images.append((f"{id2}-{ord}.png", save_img))
    return images

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

def rename(country):
    if country == "peru":
        ord = {}
        seen = set()
        with open(os.path.join(util.root(), "peru", "peru_roads.txt"), "rt") as fd:
            for line in fd:
                fname = line.strip()
                idx = os.path.splitext(fname)[0].split("_")[-1]
                ord[fname] = int(idx in seen)
                seen.add(idx)

    dir = os.path.join(util.root(), country, f"{country}_1000x1000_images")
    for fname_old in sorted(os.listdir(dir)):
        if fname_old.endswith(".tif"):
            if len(fname_old.split("_")) != 4:
                return
            idx = os.path.splitext(fname_old)[0].split("_")[-1]
            if country == "peru":
                fname_new = f"{idx}-{ord[fname_old]}.tif"
            else:
                fname_new = f"{idx}.tif"
            os.rename(os.path.join(dir, fname_old), os.path.join(dir, fname_new))
