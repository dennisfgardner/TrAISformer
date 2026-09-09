#!/usr/bin/env python3

"""evaluate the model

"""

import copy
import pickle
from pathlib import Path

import torch
import numpy as np
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
from torch.nn import functional as F
from shapely.geometry import Polygon

import models
from config_trAISformer import Config

# region of interest (ROI) of the dataset selected in Config
CF = Config()
LAT_MIN = CF.lat_min
LAT_MAX = CF.lat_max
LON_MIN = CF.lon_min
LON_MAX = CF.lon_max

# the cached raster only covers one ROI, so key it by dataset
BASEMAP_PATH = Path(f"basemap_{CF.dataset_name}.tif")


def save_basemap():
    """save the map raster"""
    cx.bounds2raster(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, ll=True,
                     path=str(BASEMAP_PATH), source=cx.providers.OpenTopoMap)


def basemap():
    """get basemap"""

    # only need to run once to generate local raster
    if not BASEMAP_PATH.exists():
        save_basemap()

    lon_point_list = [LON_MIN, LON_MIN, LON_MAX, LON_MAX]
    lat_point_list = [LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]

    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    gdf = gpd.GeoDataFrame(geometry=[polygon_geom], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot the GeoDataFrame
    gdf.boundary.plot(
        ax=ax,
        aspect='equal',
        alpha=1.0,
        linewidth=1.5,
        color='black'
    )

    # Add the basemap using contextily
    cx.add_basemap(ax, crs=gdf.crs, source=str(BASEMAP_PATH), alpha=1.0)

    return fig, ax


def model_size(model):
    """count model parameters and size"""
    model_num_params = 0
    param_size = 0
    for param in model.parameters():
        model_num_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024**2
    return model_num_params, model_size_mb


def get_test_data():
    """get test data

    - data is a list of dicts, dicts with keys "mmsi" and "traj"
    - trajectory data is [lat, lon, sog, cog, unix_timestamp, mmsi]
    - data is normalized, to convert to original (lat, lon) the max/min values
      are needed from the from author's paper
    """

    datapath = Path(CF.datadir) / CF.testset_name
    with open(datapath, "rb") as f:
        # raw data is a list of dicts with keys "mmsi" and "traj"
        data = pickle.load(f)
    return data


def undo_norm_ll(norm_lats, norm_lons):
    """take normalized lat and lon and convert back to original unit"""
    orig_lats = norm_lats * (LAT_MAX-LAT_MIN) + LAT_MIN
    orig_lons = norm_lons * (LON_MAX - LON_MIN) + LON_MIN
    return orig_lats, orig_lons


def main():
    """main entry point"""

    # config is struct holding variables
    cf = CF

    # load the model
    model = models.TrAISformer(cf, partition_model=None)
    model.to(cf.device)
    model.load_state_dict(torch.load(cf.ckpt_path))

    # model size calculation
    model_num_params, model_size_mb = model_size(model)
    print(f"model parameters: {model_num_params:,}")
    print(f'model size: {model_size_mb:.3f}MB')

    # load test data
    all_tracks = get_test_data()
    print(f"loaded {len(all_tracks)} tracks")
    # pick a single track and un-normalize the (lat, lon) coords
    track_num = 0
    track = all_tracks[track_num]["traj"]
    norm_lats = track[:, 0]
    norm_lons = track[:, 1]
    lats, lons = undo_norm_ll(norm_lats, norm_lons)

    # pick half the track to make predictions
    sub_seq = track[:int(track.shape[0]/2), 0:4]
    seq = sub_seq[np.newaxis, :]
    seq = torch.tensor(seq).to(cf.device)

    for ii in range(0, 15):
        print(ii)

        # predict the next point
        model.eval()
        with torch.no_grad():
            logits, _ = model(seq)
            # "pluck the logits at the final step" - orig. author
            logits = logits[:, -1, :]
            lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
                logits,
                (model.lat_size, model.lon_size,
                    model.sog_size, model.cog_size),
                dim=-1
            )
        # convert logits to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)
        # sample from distribution
        lat_ix = torch.multinomial(lat_probs, num_samples=1)
        lon_ix = torch.multinomial(lon_probs, num_samples=1)
        sog_ix = torch.multinomial(sog_probs, num_samples=1)
        cog_ix = torch.multinomial(cog_probs, num_samples=1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seq.device) + 0.5
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes
        seq = torch.cat((seq, x_sample.unsqueeze(1)), dim=1)

        # visualization
        sub_lats, sub_lons = undo_norm_ll(sub_seq[:, 0], sub_seq[:, 1])
        lat_probs = lat_probs[:].detach().clone().cpu().numpy()
        lon_probs = lon_probs[:].detach().clone().cpu().numpy()

        X, Y = np.meshgrid(lon_probs, lat_probs)
        prob = X*Y
        # get a copy of the gray color map
        my_cmap = copy.copy(plt.get_cmap('jet'))
        # set how the colormap handles 'bad' values
        my_cmap.set_bad(alpha=0)

        threshold = 0.001
        # insert 'bad' values into data
        prob[prob < threshold] = np.nan

        _, ax = basemap()
        ax.plot(lons, lats, color="red", linewidth=2, zorder=1,
                label="True Path")
        ax.scatter(sub_lons, sub_lats, color="black", marker=".", s=100,
                   zorder=2)
        ax.scatter(lons[0], lats[0], color="green", marker="8", s=100,
                   zorder=3)
        ax.scatter(lons[-1], lats[-1], color="red", marker="X", s=100,
                   zorder=4)
        ax.imshow(prob, origin="lower",
                  extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), cmap=my_cmap)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    main()
