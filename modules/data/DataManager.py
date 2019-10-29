import os

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import modules

class DataManager:
    
    def __init__(self, config):
        self.config = config
        
        self.filenames = {}
        self.dataframes = {}
        self.shapefiles = {}
        
        self._setup_countries = set()
        
        if config["use_kenya_images"]:
            self._setup("kenya")
        
        if config["use_peru_images"]:
            self._setup("peru")


    def _setup(self, country):
        if country != "kenya" and country != "peru":
            raise ValueError("Country must be either \'kenya\' or \'peru\'.")

        if country not in self._setup_countries:
            geo = modules.data.load_geodata(country)
            osm, sf = modules.data.load_shapefile(country)

            self.shapefiles[country] = sf
            self.dataframes[country] = pd.DataFrame.merge(geo, osm, on="index")

            classes = [self.config["class_enum"][v] for v in self.dataframes[country]["class"].values]
            self.dataframes[country]["label"] = classes
            
            # indices only needed if need to 
            # self.indices[country] = set(modules.data.util.load_image_indices(country))

            self._setup_countries.add(country)


    def class_weight(self, country):
        class_weight = None

        if self.config["balance_classes"]:
            class_weight = compute_class_weight(
                "balanced", np.arange(self.config["n_classes"]), self.dataframes[country]["label"].values
            )

        return class_weight

    
    def generate_kenya(self):
        preprocessing_function = None
        if self.config["pretrained"]:
            module = modules.models.pretrained_cnn_module(self.config["pretrained"]["type"])
            preprocessing_function = getattr(module, "preprocess_input")
            
            datagen = ImageDataGenerator(
                preprocessing_function=preprocessing_function,
                validation_split=self.config["validation_split"],
            )
            datagen2 = ImageDataGenerator(
#                 preprocessing_function=preprocessing_function,
                validation_split=self.config["validation_split"],
            )
            
            directory = f"{modules.data.util.root()}/kenya/{self.config['image_size']}/{self.config['resizing']}"
            
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist. Falling back to concurrent resizing.")
                directory = f"{modules.data.util.root()}/kenya/kenya_1000x1000_images"
            
            dataframe = pd.DataFrame(
                list(map(
                    lambda e: (f"{e[0]}_{e[1]}.jpg", e[2]), 
                    zip(
                        self.dataframes['kenya'].index, 
                        map(int, self.dataframes['kenya']["id"]),
                        self.dataframes['kenya']["class"]
                    )
                )),
                columns=["filename", "class"]
            )
            
            if self.config["sample"]:
                dataframe = dataframe.sample(n=self.config["sample"]["size"], replace=False, random_state=self.config["seed"])
                
            if self.config["mask"] == 'none':
                train_generator = datagen.flow_from_dataframe(
                    dataframe,
                    directory=directory, 
                    subset="training",
                    class_mode='categorical',
                    batch_size=self.config["batch_size"],
                    seed=self.config["seed"],
                    shuffle=self.config["shuffle"],
                    target_size=(self.config["image_size"], self.config["image_size"])
                )

                val_generator = datagen.flow_from_dataframe(
                    dataframe,
                    directory=directory, 
                    subset="validation",
                    class_mode='categorical',
                    batch_size=self.config["batch_size"],
                    seed=self.config["seed"],
                    shuffle=self.config["shuffle"],
                    target_size=(self.config["image_size"], self.config["image_size"])
                )
            elif self.config["mask"] == 'overlay':
                raise NotImplementedError()
            elif self.config["mask"] == 'occlude':
                mask_df = pd.DataFrame(list(map(
                    lambda e: (f"{e[0]}_kenya_224x224_mask_10.png", e[2]), 
                    zip(
                        self.dataframes['kenya'].index, 
                        map(int, self.dataframes['kenya']["id"]),
                        self.dataframes['kenya']["class"]
                    )
                    )),
                columns=["filename", "class"])

                mask_df = mask_df.iloc[dataframe.index]
                mask_directory = f"{modules.data.util.root()}/kenya/kenya_224x224_masks_10/"
                train_generator=self.multiple_generator(datagen, datagen2, dataframe, mask_df, directory, mask_directory, 'training')
                val_generator=self.multiple_generator(datagen, datagen2, dataframe, mask_df, directory, mask_directory, 'validation')             
        else:
            raise NotImplementedError("Custom model and preprocessing pipeline not yet defined.")
        return train_generator, val_generator


    def generate_peru(self):
        raise NotImplementedError()
        
        
    def multiple_generator(self, generator, generator_no_preprocess, img_df, mask_df, img_dir, mask_dir, subset):
        img_generator = generator.flow_from_dataframe(
            img_df,
            directory=img_dir, 
            subset=subset,
            class_mode='categorical',
            batch_size=self.config["batch_size"],
            seed=self.config["seed"],
            shuffle=self.config["shuffle"],
            target_size=(self.config["image_size"], self.config["image_size"])
        )
        mask_generator = generator_no_preprocess.flow_from_dataframe(
            mask_df,
            directory=mask_dir, 
            subset=subset,
            class_mode='categorical',
            batch_size=self.config["batch_size"],
            seed=self.config["seed"],
            shuffle=self.config["shuffle"],
            target_size=(self.config["image_size"], self.config["image_size"])
        )
        while True:
            img, label = img_generator.next()
            mask, _ = mask_generator.next()
            print(img)
            print(mask)
            print(np.sum(mask))
            print(np.max(mask))
            # np.flip to account for upside-down mask
            yield img.astype(np.float32), (img * np.flip(mask, axis=1)).astype(np.float32), label
