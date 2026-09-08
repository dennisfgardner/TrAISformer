#!/usr/bin/env python3

"""view the data

TODO:
- load test data
- look at data
- plot data on map
- make a prediction and plot on map
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from config_trAISformer import Config

# region of interest (ROI) of the dataset selected in Config
CF = Config()
LAT_MIN = CF.lat_min
LAT_MAX = CF.lat_max
LON_MIN = CF.lon_min
LON_MAX = CF.lon_max

DATADIR = Path(CF.datadir)
# only shipped with ct_dma; other datasets have no coastline file
COASTLINE_NAME = "dma_coastline_polygons.pkl"


def view_coastline_data():
    """view coastline data

    - data is a list of numpy arrays, each (lat, lon) coords
    - data is not normalized
    """
    path = DATADIR / COASTLINE_NAME
    if not path.exists():
        print(f"no coastline file at {path}, skipping")
        return
    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    _, ax = plt.subplots()
    for item in raw_data:
        ax.scatter(item[:, 1], item[:, 0])


def view_test_data():
    """view test data

    - data is a list of dicts, dicts with keys "mmsi" and "traj"
    - data is normalized, to convert to original (lat, lon) the max/min values
      of the ROI are needed, taken from Config
    """

    datapath = DATADIR / CF.testset_name
    with open(datapath, "rb") as f:
        # raw data is a list of dicts with keys "mmsi" and "traj"
        raw_data = pickle.load(f)
    print(f"there are {len(raw_data)} trajectories")

    _, ax = plt.subplots()
    for item in raw_data:
        norm_lats = item['traj'][:, 0]
        orig_lats = norm_lats * (LAT_MAX-LAT_MIN) + LAT_MIN

        norm_lons = item['traj'][:, 1]
        orig_lons = norm_lons * (LON_MAX - LON_MIN) + LON_MIN

        ax.scatter(orig_lons, orig_lats)


def main():
    """main entry point"""


if __name__ == "__main__":
    view_coastline_data()
    view_test_data()
    plt.show()
