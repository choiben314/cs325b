#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tfcle

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from modules.run import load_config, Trainer, Metrics
from modules.data import DataManager, processing, util
from modules.models import pretrained_cnn, pretrained_cnn_multichannel


# In[2]:


def preprocess(country, config, data_manager):
    print(f"Preprocess {country}...")

    print()

    print("+=+ RENAMING IMAGES.... +=+")
    processing.rename(country)

    print("+=+ FILTERING IMAGES... +=+")
    processing.generate_filters(country)

    print("+=+ DOWNCROPPING IMAGES +=+")
    processing.downcrop(country, config["image_size"])

    print("+=+ INIT DATA MANAGER.. +=+")
    data_manager._setup(country)

    print("+=+ GENERATING MASKS... +=+")
    if country is "kenya":
        processing.generate_masks(country, config, data_manager)

    print()


# In[3]:


def run_experiment_from_config(config_file, country):

    config = load_config(config_file)
    data_manager = DataManager(config)

    class_weight = None
    train_generator = None
    val_generator = None

    if country == 'kenya':
        train_generator, val_generator, dataframe = data_manager.generate_kenya()
        class_weight = data_manager.class_weight("kenya")
    elif country == 'peru':
        train_generator, val_generator, dataframe = data_manager.generate_peru()
        class_weight = class_weight=[1.64, 1, 2]
    
    convnet = pretrained_cnn_multichannel(config, image_size=config["image_size"], n_channels=config["n_channels"])

    val_steps = config["sample"]["size"] * (config["validation_split"]) // config["batch_size"] + 1

    labels = None
    if config['mask'] is not None:
        epochs = 0
        labels = []
        for data, label in val_generator:
            if epochs >= val_steps:
                break
            labels.extend(np.argmax(label, axis=1))
            epochs += 1
        labels = np.array(labels)
    trainer = Trainer(config)
    metrics_callback = Metrics(val_generator, trainer.tensorboard_dir, labels, val_steps)
    trainer.callbacks.append(metrics_callback)

    convnet.compile(loss=trainer.loss, optimizer=trainer.optimizer, metrics=config["weighted_metrics"])

    convnet.fit_generator(
        train_generator, 
        config["sample"]["size"] * (1 - config["validation_split"]) // config["batch_size"] + 1,
        epochs=config["n_epochs"],
        callbacks=trainer.callbacks,
        validation_data=val_generator, 
        validation_steps=val_steps,
        class_weight=class_weight,
        use_multiprocessing=True
    )


# In[4]:


def setup_cross_domain_from_config(config_file, country):

    config = load_config(config_file)
    data_manager = DataManager(config)

    class_weight = None
    train_generator = None
    val_generator = None

    if country == 'kenya':
        train_generator, val_generator, dataframe = data_manager.generate_kenya()
    elif country == 'peru':
        train_generator, val_generator, dataframe = data_manager.generate_peru()
    
    convnet = pretrained_cnn_multichannel(config, image_size=config["image_size"], n_channels=config["n_channels"])
    return convnet, val_generator

def best_weights(directory):
    fnames = [fname for fname in os.listdir(directory) if fname.endswith("hdf5")]
    fname = min(fnames, key=lambda fname: float(fname.split("-")[-1].split(".")[0]))
    return fname


# ### Preprocess the data

# In[ ]:


config = load_config("preprocess")

# Initialize the DataManager with no data
config["use_kenya_images"] = False
config["use_peru_images"] = False
data_manager = DataManager(config)

preprocess("kenya", config, data_manager)
preprocess("peru", config, data_manager)


# ### Run the Experiments

# In[ ]:


# Xception and Masking Experiments
run_experiment_from_config("final_xception_kenya_rgb", "kenya")
run_experiment_from_config("final_xception_peru_rgb", "peru")
run_experiment_from_config("final_xception_kenya_masked", "kenya")
run_experiment_from_config("final_xception_kenya_masked-inverted", "kenya")
run_experiment_from_config("final_xception_kenya_two_with_mask", "kenya")

# ResNetV2 and Binarization Experiments
run_experiment_from_config("final_resnet_kenya", "kenya")
run_experiment_from_config("final_resnet_kenya_balanced", "kenya")
run_experiment_from_config("final_resnet_kenya_balanced_major_vs_all", "kenya")
run_experiment_from_config("final_resnet_kenya_balanced_major_vs_minor", "kenya")
run_experiment_from_config("final_resnet_kenya_balanced_major_vs_twotrack", "kenya")
run_experiment_from_config("final_resnet_kenya_balanced_minor_vs_all", "kenya")
run_experiment_from_config("final_resnet_kenya_balanced_minor_vs_twotrack", "kenya")


# ### Analysis

# In[ ]:


o_file = open(os.path.join(util.root(), "..", "submission_out.txt"), "wt")
to_write = ""


# In[ ]:


try:
    name = "final_xception_peru_balanced"
    path = os.path.join(util.root(), "data", name)
    model, val_gen = setup_cross_domain_from_config(name, "kenya")
    model.load_weights(os.path.join(path, best_weights(path)))
    val_predict = np.argmax(model.predict(val_gen), axis=1)

    to_write += f"acc(peru -> kenya): {str(accuracy_score(val_gen.classes, val_predict))}\n"
    average = "macro"
    to_write += f"f_1(peru -> kenya): {str(f1_score(val_gen.classes, val_predict, average=average))}\n"
except:
    to_wrote += "WARNING: failed to record (peru -> kenya) metrics"


# In[ ]:


try:
    name = "final_xception_kenya_balanced"
    path = os.path.join(util.root(), "data", name)
    model, val_gen = setup_cross_domain_from_config(name, "peru")
    model.load_weights(os.path.join(path, best_weights(path)))
    val_predict = np.argmax(model.predict(val_gen), axis=1)
    
    to_write += f"acc(kenya -> peru): {str(accuracy_score(val_gen.classes, val_predict))}\n"
    average = "macro"
    to_write += f"f_1(kenya -> peru): {str(f1_score(val_gen.classes, val_predict, average=average))}\n"
except:
    to_wrote += "WARNING: failed to record (kenya -> peru) metrics"


# In[ ]:


to_write += "For detailed training and validation metrics for all experiments"   +             "please see the Tensorboard event files in the root/data/<config>/" +             "tensorboard directory."

o_file.write(to_write)
o_file.close()


# In[ ]:




