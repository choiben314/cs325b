import os
import yaml


def load_config(base, tag=None):
    assert base.startswith("cls") or base.startswith("seg")       
    
    if tag is not None:
        base = f"{base}_{tag}"
    path_to_def = os.path.join(os.path.dirname(__file__), "config", f"{base}.yaml")
    
    with open(path_to_def, "rt") as def_file:
        def_dict = yaml.load(def_file, Loader=yaml.FullLoader)
    
    return def_dict