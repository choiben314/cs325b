import os

def root():
    return f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/data"

def cache_image_filenames(country, D):
    validate_country(country)
    
    path = os.path.join(root(), country, f"{country}_{D}x{D}_images.txt")
    
    with open(path, "wt") as ofile:
        for fname in os.listdir(os.path.join(root(), country, f"{country}_{D}x{D}_images")):
            if fname.endswith(".tif") or fname.endswith(".npy"):
                ofile.write(f"{fname}\n")
                
def load_image_filenames(country):
    validate_country(country)
    
    path = os.path.join(root(), country, f"{country}_roads.txt")
    
    with open(path, "rt") as ifile:
        fnames = [fname.strip() for fname in ifile]
    
    return fnames

def lookup_filename(fnames, road_id):
    for fname in fnames:
        if road_id in fnames:
            return fname

def validate_country(country):
    if country != "peru" and country != "kenya":
        raise ValueError("Parameter \'country\' must be one of either \'kenya\' or \'peru\'.")