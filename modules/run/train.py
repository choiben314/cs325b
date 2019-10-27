import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam

from modules.run import Runner

class Trainer(Runner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        checkpoints = True
        tensorboard = True
        
        self.init_directories(checkpoints, tensorboard)
        
        self.init_callbacks(checkpoints, tensorboard)
        
        self.init_optimizer()
        
    def init_callbacks(self, checkpoints=True, tensorboard=True):
        self.tensorboard_callback = TensorBoard(
            self.tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq=self.config["batch_size"] * self.config["tensorboard_freq"]
        )

        self.checkpoints_callback = ModelCheckpoint(
            os.path.join(self.checkpoints_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch',
        )
        
        self.callbacks = [self.tensorboard_callback, self.checkpoints_callback]
    
    def init_optimizer(self):
        if self.config["optimizer"] == "sgd":
            self.optimizer = SGD(learning_rate=self.config["learning_rate"])
        elif self.config["optimizer"] == "adam":
            self.optimizer = Adam(learning_rate=self.config["learning_rate"])
        else:
            raise ValueError