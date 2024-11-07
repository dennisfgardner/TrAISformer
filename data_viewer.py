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


def view_coastline_data():
    """view coastline data

    - data is a list of numpy arrays, each (lat, lon) coords
    - data is not normalized
    """
    path = Path("data/ct_dma/dma_coastline_polygons.pkl")
    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    _, ax = plt.subplots()
    for item in raw_data:
        ax.scatter(item[:, 1], item[:, 0])


def view_test_data():
    """view test data

    - data is a list of dicts, dicts with keys "mmsi" and "traj"
    - data is normalized, to convert to original (lat, lon) the max/min values
      are needed from the from author's paper
    """

    datapath = Path("data/ct_dma/ct_dma_test.pkl")
    with open(datapath, "rb") as f:
        # raw data is a list of dicts with keys "mmsi" and "traj"
        raw_data = pickle.load(f)
    print(f"there are {len(raw_data)} trajectories")

    # from author's paper
    LAT_MIN = 55.5
    LAT_MAX = 58.0
    LON_MIN = 10.3
    LON_MAX = 13.0
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
