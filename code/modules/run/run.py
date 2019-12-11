import os
import yaml

from tensorflow.keras.losses import CategoricalCrossentropy
from modules.data import util


def load_config(fname):
    path_to_def = os.path.join(util.root(), "config", f"{fname}.yaml")
    
    with open(path_to_def, "rt") as def_file:
        def_dict = yaml.load(def_file, Loader=yaml.FullLoader)
    
    return def_dict


class Runner:
    
    def __init__(self, config):
        self.config = config
        
        self.init_loss()
        
    def init_directories(self, checkpoints=True, tensorboard=True):
        checkpoints_dir = os.path.join(util.root(), self.config["name"], "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir

        tensorboard_dir = os.path.join(util.root(), self.config["name"], "tensorboard")
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.tensorboard_dir = tensorboard_dir
        
    def init_loss(self):
        self.loss = CategoricalCrossentropy()
            
    

        
        


