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
        self.directories = {}
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
            
            root = modules.data.util.root()
            resizing = self.config['resizing'] if self.config['resizing'] != "none" else ""
            self.directories[country] = os.path.join(root, country, str(self.config['image_size']), resizing)
            
            valid = self.file_is_valid(self.dataframes[country], self.directories[country])
            self.dataframes[country] = self.dataframes[country][valid]

            self._setup_countries.add(country)

    def file_is_valid(self, dataframe, directory):
        filenames = self._extract_filenames(directory)
        valid = []
        for i in dataframe.index:
            if self.config["resizing"] == "none":
                fname = self._format_filename(i, int(dataframe['id'].loc[i]), ext="tif")
            else:
                fname = self._format_filename(i, int(dataframe['id'].loc[i]), ext="jpg")
            valid.append(fname in filenames)
        return np.array(valid)

    def class_weight(self, dataframe, country):
        class_weight = None

        if self.config["weight_classes"]:
            classes = np.arange(self.config["n_classes"])
            weights = compute_class_weight("balanced", classes, dataframe["class"].values.astype(np.int32))
            class_weight = {cls:weight for cls, weight in zip(classes, weights)}

        return class_weight
    
    def sample_class(self, dataframe, cls, n):           
        df = dataframe[dataframe["class"] == cls]
        return df.sample(n=n, replace=False, random_state=self.config["seed"])
            
    def generate_kenya(self):
        
        # get input directory
        directory = self.directories["kenya"]
        
        # format dataframe for ImageDataGenerator.flow_from_dataframe
        dataframe = self._format_dataframe_for_flow("kenya")
                
        if self.config['remove_clouds']:
            cloud_directory = f"{modules.data.util.root()}/kenya/cloudy.txt"
            cloud_filenames = pd.read_csv(cloud_directory, sep=" ", header=None)
            cloud_filenames.columns = ["filename"]
            
            cloud_filenames["filename"] = cloud_filenames.filename.str.slice(16)
            cloud_filenames["filename"] = cloud_filenames.filename.str.slice(0, -4) + ".jpg"

            dataframe = dataframe[~dataframe['filename'].isin(cloud_filenames.filename)]
            
            print("Declouded dataframe length: " + str(len(dataframe.index)))

        # cull unused classes from data
        labels = set()
        for cls in self.config["class_enum"]:
            if self.config["class_enum"][cls] >= 0:
                labels.add(str(self.config["class_enum"][cls]))
        dataframe = dataframe[np.isin(dataframe["class"], list(labels))]
        
        # sample the data
        if "sample" in self.config:                 
            if not self.config["sample"]["balanced"]:
                dataframe = dataframe.sample(n=self.config["sample"]["size"], replace=False, random_state=self.config["seed"])
            else:
                n = self.config["sample"]["size"] // len(labels)
                dataframe = pd.concat([self.sample_class(dataframe, label, n) for label in labels])
                
        # shuffle the data
        dataframe = dataframe.reindex(np.random.permutation(dataframe.index))        

        # define data preprocessing
        preprocessing_function = None
        if self.config["pretrained"]:
            module = modules.models.pretrained_cnn_module(self.config["pretrained"]["type"])
            preprocessing_function = getattr(module, "preprocess_input")
        else:
            raise NotImplementedError("Custom model and preprocessing pipeline not yet defined.")
        
        # make image data generator for rgb
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            validation_split=self.config["validation_split"],
        )
        
        if self.config["mask"] == 'none':
            train_generator = self._build_generator(datagen, dataframe, directory, "training")
            val_generator = self._build_generator(datagen, dataframe, directory, "validation")            
        elif self.config["mask"] == "occlude" or self.config["mask"] == "overlay":
            datagen_mask = ImageDataGenerator(validation_split=self.config["validation_split"])
            dataframe_mask = pd.DataFrame(
                list(map(
                    lambda e: (f"{e[0]}_kenya_224x224_mask_20.png", e[2]), 
                    zip(
                        self.dataframes['kenya'].index,
                        map(int, self.dataframes['kenya']["id"]),
                        self.dataframes['kenya']["label"]
                    )
                )),
                columns=["filename", "class"])
            dataframe_mask = dataframe_mask.iloc[dataframe.index]
            directory_mask = f"{modules.data.util.root()}/kenya/kenya_224x224_masks_20/"

            train_generator = self.multiple_generator(
                datagen, datagen_mask, 
                dataframe, dataframe_mask, 
                directory, directory_mask, 
                'training'
            )
            val_generator = self.multiple_generator(
                datagen, datagen_mask, 
                dataframe, dataframe_mask, 
                directory, directory_mask, 
                'validation'
            )
               
        return train_generator, val_generator, dataframe


    def generate_peru(self):
        ##
        #
        # TODO: build out peru logic
        #
        ##
        raise NotImplementedError()
        
        
    def multiple_generator(self, datagen1, datagen2, dataframe1, dataframe2, directory1, directory2, subset):
        generator1 = self._build_generator(datagen1, dataframe1, directory1, subset)
        generator2 = self._build_generator(datagen2, dataframe2, directory2, subset)
        while True:
            x1, y1 = generator1.next()
            x2, y2 = generator2.next()
            if self.config["mask"] == "occlude":
                if not self.config['mask_inverted']:
                    yield (x1 * np.flip(x2, axis=1)).astype(np.float32), y1
                else:
                    yield (x1 * (1 - np.flip(x2, axis=1))).astype(np.float32), y1
            elif self.config["mask"] == "overlay":
                yield np.concatenate((x1, np.expand_dims(np.flip(x2, axis=1)[:, :, :, 0], axis=3)), axis=3), y1
                
    def _build_generator(self, datagen, dataframe, directory, subset):
        return datagen.flow_from_dataframe(
            dataframe,
            directory=directory, 
            subset=subset,
            class_mode='categorical',
            batch_size=self.config["batch_size"],
            seed=self.config["seed"],
            shuffle=self.config["shuffle"],
            target_size=(self.config["image_size"], self.config["image_size"])
        )
            
    def _format_dataframe_for_flow(self, country):
        if self.config["resizing"] == "none":
            ext = "tif"
        else:
            ext = "jpg"
        return pd.DataFrame(
            list(map(
                lambda e: (self._format_filename(e[0], e[1], ext=ext), e[2]), 
                zip(
                    self.dataframes[country].index, 
                    map(int, self.dataframes[country]["id"]),
                    map(str, self.dataframes[country]["label"]),
                )
            )),
            columns=["filename", "class"]
        )
            
    def _format_filename(self, id1, id2, ext):
        return f"{id1}_{id2}.{ext}"
    
    def _extract_filenames(self, directory):
        filenames = map(lambda x: x.strip(), os.listdir(directory))
        return set(filenames)
