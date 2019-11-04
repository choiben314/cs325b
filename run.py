import os
import tensorflow as tf

from modules.run import load_config
from modules.run import Trainer
from modules.data import DataManager
from modules.models import pretrained_cnn

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import cv2

print(tf.test.is_gpu_available())

with tf.device('device:GPU:0'):

    config = load_config("cls_w6_e2")
    data_manager = DataManager(config)
    convnet = pretrained_cnn(config, image_size=config["image_size"], n_channels=config["n_channels"])
    trainer = Trainer(config)
    train_generator, val_generator = data_manager.generate_kenya()

    convnet.compile(loss=trainer.loss, weighted_metrics=['accuracy'], optimizer=trainer.optimizer)

    convnet.summary()

    convnet.fit_generator(
        train_generator, 
        config["sample"]["size"] * (1 - config["validation_split"]) // config["batch_size"],
        epochs=config["n_epochs"],
        callbacks=trainer.callbacks, 
        validation_data=val_generator, 
        validation_steps=config["sample"]["size"] * (config["validation_split"]) // config["batch_size"],
        class_weight=data_manager.class_weight("kenya")
    )
