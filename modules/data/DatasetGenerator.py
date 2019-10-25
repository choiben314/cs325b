import tensorflow as tf
import pandas
import numpy as np

import modules


class DatasetGenerator:
    
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
            self.dataframes[country] = pandas.DataFrame.merge(geo, osm, on="index")
            
            classes = [self.config["class_enum"][v] for v in self.dataframes[country]["class"].values]
            self.dataframes[country]["label"] = classes
            
            # indices only needed if need to 
            # self.indices[country] = set(modules.data.util.load_image_indices(country))
        
            self._setup_countries.add(country)
        
        
    def sample(self, country, N, validate=True):
        
        def _sample(cls):
            df = self.dataframes[country]
            return df.iloc[np.random.choice(df[df["class"] == cls].index, size=N, replace=False)]
        
        major = _sample("major")
        minor = _sample("minor")
        two_track = _sample("two-track")
        
        df = pandas.concat([major, minor, two_track])
        
        filenames = self._extract_filenames(df)
                
        # major => 0, minor => 1, two-track => 2
        labels = np.arange(len(filenames)) // N
        labels = np.eye(np.max(labels) + 1)[labels]
        
        return filenames, labels

            
    def generate_kenya(self):
        self._setup("kenya")
        
        if self.config.__contains__("sample"):
            filenames, labels = self.sample("kenya", self.config["sample"]["size"])
        else:
            df = self.dataframes["kenya"]
            
            filenames = self._extract_filenames(df)
            
            labels = np.zeros(len(filenames))
            labels[df["class"] == "major"] = 0
            labels[df["class"] == "minor"] = 1
            labels[df["class"] == "two-track"] = 2
            
            labels = np.eye(np.max(labels) + 1)[labels]

        dataset = tf.data.Dataset.from_generator(
            lambda: ((self._load_filename("kenya", filename), label) for filename, label in zip(filenames, labels)),
            output_types=(tf.int32, tf.int32),
            output_shapes=((self.config["image_size"], self.config["image_size"], 3), (labels.shape[1], ))
        )
        
        if self.config["shuffle_buffer"]:
            dataset = dataset.shuffle(self.config["shuffle_buffer"])
        
        dataset = dataset.batch(self.config["batch_size"], drop_remainder=True)
        
        return dataset
        
    
    def generate_peru(self):
        raise NotImplementedError()
    
    
    def _extract_filenames(self, df):
        filenames = []
        for idx in df.index:
            filenames.append("{}_{}.npy".format(idx, int(df.loc[idx]['id'])))
        return filenames

    def _load_filename(self, country, filename):
        return np.load(os.path.join("data", country, f"{country}_{self.config['image_size']}x{self.config['image_size']}_images", filename))