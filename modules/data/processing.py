import cv2
import numpy as np

def rescale(country, filename, w, h):
    validate_country(country)
    
    path = os.path.join("data", f"{country}_1000x1000_images", filename)
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist.")
    
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if image is not None:
        return skimage.transform.resize(image, (w, h), anti_aliasing=False)
    else:
        return None
    
def downscale(country, D):
    fnames = util.load_image_filenames(country, D)
    for i, fname in enumerate(fnames):
        src = os.path.join(util.root(), f"{country}_1000x1000_images", fname)
        if os.path.exists(src):
            image = rescale("country", fname, D, D)
            if image is not None:
                tgt = os.path.join(util.root(), f"{country}_{D}x{D}_images", "_".join(fname.split(".")[0].split("_")[-2:]))
                np.save(tgt, (image * 255).astype('uint8'))