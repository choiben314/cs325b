import os
import json
import pandas
import shapefile

from modules.data import util

CLASSMAP = {
    "motorway": "major",
    "trunk": "major",
    "primary": "major",
    "secondary": "major",
    "tertiary": "minor",
    "unclassified": "minor",
    "motorway_link": "major",
    "trunk_link": "major",
    "primary_link": "major",
    "secondary_link": "major",
    "tertiary_link": "minor",
    "track": "two-track"
}

# .dbf/.shp polyline and class handling data

def _load_shapefile(country, encoding="iso-8859-1"):
    util.validate_country(country)
    dbf = open(os.path.join(util.root(), country, f"{country}_roads.dbf"), "rb")
    shp = open(os.path.join(util.root(), country, f"{country}_roads.shp"), "rb")
    shx = open(os.path.join(util.root(), country, f"{country}_roads.shx"), "rb")
    sf = shapefile.Reader(shp=shp, dbf=dbf, shx=shx)
    sf.encoding = encoding
    return sf

def load_shapefile(country):
    util.validate_country(country)
    sf = _load_shapefile(country, encoding="iso-8859-1")
    table = []
    columns = ("index", "highway", "class", "name")
    for i, sr in enumerate(sf.shapeRecords()):
        if sr.record.highway in CLASSMAP:
            row = (
                i,
                sr.record.highway,
                CLASSMAP[sr.record.highway],
                sr.record.name
            )
            table.append(row)
    df = pandas.DataFrame.from_records(table, index="index", columns=columns)
    return df, sf

# .csv road and image bounding box data handling

def load_geodata(country):
    util.validate_country(country)
    path = os.path.join(util.root(), country, f"{country}_roads_bbox_300m.csv")
    if country == "kenya":
        df = pandas.read_csv(path, index_col="index")
        df = df.drop(columns=["Unnamed: 0"])
        df = df.rename(lambda x: x.lower(), axis="columns")
        df = df.reindex(sorted(df.columns), axis=1)
    elif country == "peru":
        df = pandas.read_csv(path, index_col="index")
        df["lat"] = list(map(lambda x: json.loads(x)["coordinates"][1], df[".geo"]))
        df = df.drop(columns=[".geo", "system:index"])
        df = df.sort_index()
        df = df.reindex(sorted(df.columns), axis=1)
    else:
        raise ValueError("Parameter \'country\' must be one of either \'kenya\' or \'peru\'.")
    df.index -= 1
    return df
