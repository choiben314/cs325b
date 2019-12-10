import numpy as np

def curvature(points):
    radians = 0
    
    for i in range(1, len(points) - 1):
        A, B, C = points[i-1:i+2]
        a_magnitude = distance(B, C).m
        a_euclidean = np.subtract(B, C)
        a_euclidean *= np.divide(a_magnitude, np.linalg.norm(a_euclidean))
        c_magnitude = distance(A, B).m
        c_euclidean = np.subtract(A, B)
        c_euclidean *= np.divide(c_magnitude, np.linalg.norm(c_euclidean))
        cosine = np.dot(a_euclidean, c_euclidean) / (a_magnitude * c_magnitude)
        angle = np.arccos(cosine)
        radians += angle
    
    return radians

def _SIFT(image, sift, plot=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(image, None)
    if plot:
        plt.imshow(cv2.drawKeypoints(image, keypoints, outImage=np.array([])))
    return keypoints

def SIFT(images):
    keypoints = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image in images:
        kpts = _SIFT(image, sift)
        keypoints.append(kpts)
    return keypoints

def HOG(images):
    features = []
    hog = cv2.HOGDescriptor()
    for image in images:
        features.append(hog.compute(image))
    return features

def canny_edge_detection(images):
    channels = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = np.median(gray)
        channel = cv2.Canny(gray, (1/2) * median, (2) * median, apertureSize=3)[:, :, None]
        channels.append(channel)
    return np.array(channels)

def location(df, fnames):
    locations = []
    for fname in fnames:
        index = fname.split("_")[0]
        locations.append(df.loc[int(index)][["lat", "lon"]].values.astype(np.float64))
    return np.array(locations)

def channel_mean(images):
    return np.mean(images, axis=(1, 2))

def channel_variance(images):
    var = None
    for i in range(3):
        if var is None:
            var = np.var(images[:, :, :, i:i+1], axis=(1, 2))
        else:
            var = np.concatenate([var, np.var(images[:, :, :, i:i+1], axis=(1, 2))], axis=-1)
    return var

def feature_set(images, dat, fnames, prefix=None):
    """
    Extract location features and RGB channel features.
        => latitude
        => longitude
        => RGB means
        => RGB variances
        
    Return a features pandas.DataFrame.
    
    This function could be modularized but is obsolete with the
    switch to neural methods.
    """
    
    locations = location(dat, fnames)
    print("lat/lon done.")
    
    rgb_means = channel_mean(images)
    print("rgb means done.")
    rgb_variances = channel_variance(images)
    print("rgb variances done.")
    
    columns = [
        "lat",
        "lon",
        "r-mean", 
        "g-mean",
        "b-mean",
        "r-var", 
        "g-var",
        "b-var",
    ]
    
    if prefix is not None:
        columns = ["-".join((prefix, col)) for col in columns]
    
    features = np.concatenate([
        locations,
        rgb_means,
        rgb_variances,
    ], axis=-1)
    
    df = pandas.DataFrame(features, columns=columns)
    
    return df
