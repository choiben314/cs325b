import modules
import cv2
import numpy as np
from skimage import draw
import skimage
import scipy.spatial as spatial

config = modules.run.load_config('cls_w6_e1')
data_manager = modules.data.DataManager(config)

orig_dim = 1000
crop_dim = 224
threshold = 10

save_dir = './data/kenya/kenya_224x224_masks_10/'

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

points = [(i, j) for i in range(orig_dim) for j in range(orig_dim)]
point_tree = spatial.cKDTree(points)

for idx in range(len(data_manager.dataframes["kenya"])):
    shpData = data_manager.shapefiles["kenya"].shape(idx).points

    min_lat = float(data_manager.dataframes["kenya"][idx:idx+1]['minlat'])
    max_lat = float(data_manager.dataframes["kenya"][idx:idx+1]['maxlat'])
    min_lon = float(data_manager.dataframes["kenya"][idx:idx+1]['minlon'])
    max_lon = float(data_manager.dataframes["kenya"][idx:idx+1]['maxlon'])

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
    save_img = img[(orig_dim // 2 - crop_dim // 2):(orig_dim // 2 + crop_dim // 2), (orig_dim // 2 - crop_dim // 2):(orig_dim // 2 + crop_dim // 2)]
    cv2.imwrite(save_dir + str(idx) + '_kenya_224x224_mask_10.png', save_img)
