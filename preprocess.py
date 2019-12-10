import os
import sys

from modules.run import load_config
from modules.data.processing import downcrop, generate_masks
from modules.data import DataManager

config = load_config("preprocess")

def preprocess(country):
    print(f"Preprocess {country}...")
    try:
        print("downcrop")
        downcrop(country, config["image_size"])

        print("data manager")
        data_manager = DataManager(config)

        print("generate masks")
        generate_masks(country, config, data_manager)

    except BaseException as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type.__name__, fname, exc_tb.tb_lineno, e)

    print()

for country in ["kenya", "peru"]:
    preprocess(country)
