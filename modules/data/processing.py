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
        line_r, line_c = skimage.draw.line(r[i], c[i], r[i + 1], c[i + 1])
        rr.extend(line_r)
        cc.extend(line_c)

    rr = np.asarray(rr)
    cc = np.asarray(cc)

    return rr, cc

# TODO: Remove threshold magic number
def generate_masks(country, config, data_manager=None, threshold=20):

    if data_manager is None:
        data_manager = DataManager(config)

    orig_dim = 1000
    crop_dim = config["image_size"]

    o_path = os.path.join(util.root(), country, str(crop_dim), "road_masks")

    if not os.path.exists(o_path):
        os.makedirs(o_path)

    points = [(i, j) for i in range(orig_dim) for j in range(orig_dim)]
    point_tree = cKDTree(points)

    for idx in data_manager.dataframes[country].index:

        shpData = data_manager.shapefiles[country].shape(idx).points

        min_lat = float(data_manager.dataframes[country][idx:idx+1]['minlat'])
        max_lat = float(data_manager.dataframes[country][idx:idx+1]['maxlat'])
        min_lon = float(data_manager.dataframes[country][idx:idx+1]['minlon'])
        max_lon = float(data_manager.dataframes[country][idx:idx+1]['maxlon'])

        points_true = [x for x in shpData if min_lat < x[1] < max_lat and min_lon < x[0] < max_lon]

        lon = [x[0] for x in points_true]
        lat = [x[1] for x in points_true]

        lat_range = max_lat - min_lat
        lat = orig_dim * (np.array(lat) - min_lat) / lat_range

        lon_range = max_lon - min_lon
        lon = orig_dim * (np.array(lon) - min_lon) / lon_range

        lon_pixel = polygon_perimeter(lon, lat)[0]
        lat_pixel = polygon_perimeter(lon, lat)[1]

        coords = []

        lon_thick = []
        lat_thick = []

        for i in range(len(lon_pixel)):
            nearest_points_idx = point_tree.query_ball_point([lon_pixel[i], lat_pixel[i]], threshold)
            for point_idx in nearest_points_idx:
                coords.append(points[point_idx])
                lon_thick.append(points[point_idx][0])
                lat_thick.append(points[point_idx][1])

        img = np.zeros((orig_dim, orig_dim), dtype=np.uint8)
        img[lat_thick, lon_thick] = 1
        save_img = resize(img, (crop_dim, crop_dim), anti_aliasing=False)
        print(os.path.join(o_path, f"{str(idx)}.png"))
        cv2.imwrite(os.path.join(o_path, f"{str(idx)}.png"), save_img)
