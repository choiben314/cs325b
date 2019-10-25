import os
import yaml


def load_config(fname):
    assert fname.startswith("cls") or fname.startswith("seg")       
    
    root = f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/data/"
    
    path_to_def = os.path.join(root, "config", f"{fname}.yaml")
    
    with open(path_to_def, "rt") as def_file:
        def_dict = yaml.load(def_file, Loader=yaml.FullLoader)
    
    return def_dict