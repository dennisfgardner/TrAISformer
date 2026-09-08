#!/usr/bin/env python3

"""Build the visuals for a finished training run.

Reads whatever is in `Config.savedir` (training log, per-epoch plots, checkpoint) and writes four
figures back into the same directory:

    loss_curves.png       train/valid loss per epoch, with the best-valid epoch marked
    training_epochs.gif   the per-epoch trajectory plots the trainer already draws, animated
    map_true_vs_pred.png  test tracks in real coordinates: the 16-sample prediction fan vs truth
    rollout_heatmap.gif   the per-step lat/lon probability heatmap advancing along one track

Run after `python trAISformer.py` has finished:

    python make_visuals.py
"""

import os
import re
import glob
import tempfile

import numpy as np
import torch
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from torch.nn import functional as F
from shapely.geometry import Polygon
from PIL import Image

import models
import trainers  # noqa: F401 -- also forces the Agg backend via trAISformer
import utils
import eval_model
from config_trAISformer import Config

CF = Config()
LAT_MIN, LAT_MAX = CF.lat_min, CF.lat_max
LON_MIN, LON_MAX = CF.lon_min, CF.lon_max

# one AIS sample every 10 minutes, so 6 steps per hour
STEPS_PER_HOUR = 6

# heatmap animation: side of the square map window, and the gamma the peaked joint
# probability is raised to before coloring
WINDOW_DEG = 1.1
GAMMA = 0.35


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def out_path(name):
    """path of an output file inside the run's savedir"""
    return os.path.join(CF.savedir, name)


def newest_log():
    """the most recent training log in savedir"""
    logs = sorted(glob.glob(os.path.join(CF.savedir, "log_*.log")))
    if not logs:
        raise FileNotFoundError(f"no log_*.log in {CF.savedir}")
    return logs[-1]


def load_test_tracks(min_len):
    """test tracks, filtered the same way trAISformer.py filters them

    Drops the leading not-yet-moving portion, then tracks with NaNs or fewer than `min_len`
    points. Returns them sorted by how far the vessel travels within the first `min_len` points
    (i.e. the window that actually gets plotted), so callers get tracks that are going somewhere
    rather than idling in a harbour.
    """
    moving_threshold = 0.05
    tracks = eval_model.get_test_data()
    for v in tracks:
        try:
            moving_idx = np.where(v["traj"][:, 2] > moving_threshold)[0][0]
        except IndexError:
            moving_idx = len(v["traj"]) - 1  # this track will be dropped below
        v["traj"] = v["traj"][moving_idx:, :]
    tracks = [v for v in tracks
              if not np.isnan(v["traj"]).any() and len(v["traj"]) >= min_len]

    def span(v):
        lat, lon = eval_model.undo_norm_ll(v["traj"][:min_len, 0], v["traj"][:min_len, 1])
        return np.hypot(np.ptp(lat), np.ptp(lon))

    return sorted(tracks, key=span, reverse=True)


def load_model():
    """the trained model from cf.ckpt_path, in eval mode on cf.device"""
    model = models.TrAISformer(CF, partition_model=None)
    model.to(CF.device)
    model.load_state_dict(torch.load(CF.ckpt_path))
    model.eval()
    return model


def add_basemap(ax):
    """draw the cached OpenTopoMap raster on `ax`, in EPSG:4326 (degrees)"""
    if not eval_model.BASEMAP_PATH.exists():
        eval_model.save_basemap()
    lons = [LON_MIN, LON_MIN, LON_MAX, LON_MAX]
    lats = [LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]
    gdf = gpd.GeoDataFrame(geometry=[Polygon(zip(lons, lats))], crs="EPSG:4326")
    gdf.boundary.plot(ax=ax, aspect="equal", alpha=0.0, linewidth=0.0, color="black")
    cx.add_basemap(ax, crs=gdf.crs, source=str(eval_model.BASEMAP_PATH), alpha=1.0)


def zoom_to(ax, lats, lons, pad_frac=0.25):
    """zoom `ax` to the bounding box of the given coords, clipped to the ROI"""
    pad_lat = max(np.ptp(lats), 0.05) * pad_frac
    pad_lon = max(np.ptp(lons), 0.05) * pad_frac
    ax.set_xlim(max(LON_MIN, lons.min() - pad_lon), min(LON_MAX, lons.max() + pad_lon))
    ax.set_ylim(max(LAT_MIN, lats.min() - pad_lat), min(LAT_MAX, lats.max() + pad_lat))


def zoom_window(ax, lats, lons, size_lat, size_lon):
    """centre a fixed-size window on the given coords, slid (not clipped) to stay in the ROI

    Panels drawn this way all come out the same shape, which a grid of subplots needs.
    """
    def window(vals, lo_v, hi_v, size):
        centre = (vals.min() + vals.max()) / 2
        lo = min(max(centre - size / 2, lo_v), hi_v - size)
        return lo, lo + size

    ax.set_ylim(*window(lats, LAT_MIN, LAT_MAX, size_lat))
    ax.set_xlim(*window(lons, LON_MIN, LON_MAX, size_lon))


def save_gif(frame_paths, dest, duration, width):
    """write an animated GIF from image files, downscaled to `width` px"""
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        frames.append(img)
    # quantize every frame against the first frame's palette so colors stay stable
    first = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    rest = [f.quantize(palette=first) for f in frames[1:]]
    first.save(dest, save_all=True, append_images=rest, duration=duration, loop=0,
               optimize=True)
    print(f"wrote {dest} ({len(frames)} frames, {os.path.getsize(dest) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 1. loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves():
    """train/valid loss per epoch, parsed out of the training log"""
    text = open(newest_log()).read()
    train = {int(e): float(v) for e, v in
             re.findall(r"Training, epoch (\d+), loss (-?[\d.]+),", text)}
    valid = {int(e): float(v) for e, v in
             re.findall(r"Valid, epoch (\d+), loss (-?[\d.]+)\.", text)}
    if not valid:
        raise RuntimeError("no epochs found in the training log")

    epochs = sorted(valid)
    v_losses = [valid[e] for e in epochs]
    best_epoch = epochs[int(np.argmin(v_losses))]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.plot(sorted(train), [train[e] for e in sorted(train)], "-o", markersize=3,
            color="#2b6cb0", label="training loss")
    ax.plot(epochs, v_losses, "-o", markersize=3, color="#c53030", label="validation loss")
    ax.axvline(best_epoch, color="#2f855a", linestyle="--", linewidth=1.2)
    ax.annotate(f"best valid loss: {min(v_losses):.3f}\n(epoch {best_epoch}, saved as model.pt)",
                xy=(best_epoch, min(v_losses)), xytext=(8, 24), textcoords="offset points",
                color="#2f855a", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2f855a", linewidth=1))
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (sum of 4 channel CE + blur NLL)")
    ax.set_title(f"TrAISformer on {CF.dataset_name}: training vs validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    dest = out_path("loss_curves.png")
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest} (best epoch {best_epoch}, valid {min(v_losses):.4f})")


# ---------------------------------------------------------------------------
# 2. per-epoch trajectory animation
# ---------------------------------------------------------------------------

def make_epoch_gif():
    """animate the epoch_XXX.jpg plots the trainer writes each epoch"""
    frames = sorted(glob.glob(os.path.join(CF.savedir, "epoch_*.jpg")))
    if not frames:
        raise FileNotFoundError(f"no epoch_*.jpg in {CF.savedir}")
    save_gif(frames, out_path("training_epochs.gif"), duration=400, width=900)


# ---------------------------------------------------------------------------
# 3. true vs predicted, on the map
# ---------------------------------------------------------------------------

def rollout(model, track, init_seqlen, steps):
    """sample CF.n_samples futures for one track; returns (preds, per-step km error)"""
    traj = track["traj"][:init_seqlen + steps, 0:4]
    seq = torch.tensor(traj[np.newaxis, :], dtype=torch.float32).to(CF.device)
    preds = trainers.sample(model, seq[:, :init_seqlen, :].repeat(CF.n_samples, 1, 1), steps,
                            temperature=1.0, sample=True,
                            sample_mode=CF.sample_mode,
                            r_vicinity=CF.r_vicinity,
                            top_k=CF.top_k)

    # haversine error of each sample against the truth, in km
    v_ranges = torch.tensor([model.lat_range, model.lon_range, 0, 0]).to(CF.device)
    v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0, 0]).to(CF.device)
    truth = seq.repeat(CF.n_samples, 1, 1)
    d = utils.haversine((truth * v_ranges + v_roi_min) * np.pi / 180,
                        (preds * v_ranges + v_roi_min) * np.pi / 180)
    return preds.cpu().numpy(), d[:, init_seqlen:].cpu().numpy()


def plot_map_true_vs_pred(n_candidates=24, hours=4):
    """the 16-sample prediction fan against ground truth, over the basemap

    Rolls out `n_candidates` moving test tracks, ranks them by mean error at the full horizon,
    then plots four spanning that range — so the figure shows what typical and bad look like,
    not just whichever tracks happened to come first.
    """
    model = load_model()
    init_seqlen = CF.init_seqlen
    steps = hours * STEPS_PER_HOUR
    candidates = load_test_tracks(min_len=init_seqlen + steps + 1)[:n_candidates]
    print(f"rolling out {len(candidates)} candidate tracks, {hours}h horizon")

    results = [(track, *rollout(model, track, init_seqlen, steps)) for track in candidates]
    results.sort(key=lambda r: r[2][:, -1].mean())

    picks = [(0, f"best of {len(results)}"),
             (len(results) // 3, "33rd percentile"),
             (2 * len(results) // 3, "67th percentile"),
             (len(results) - 1, f"worst of {len(results)}")]

    # a window big enough for every panel, so the four subplots share one shape
    coords = []
    for rank, _ in picks:
        track, preds_np, _ = results[rank]
        traj = track["traj"][:init_seqlen + steps, 0:4]
        true_lat, true_lon = eval_model.undo_norm_ll(traj[:, 0], traj[:, 1])
        pred_lat, pred_lon = eval_model.undo_norm_ll(preds_np[:, init_seqlen:, 0],
                                                     preds_np[:, init_seqlen:, 1])
        coords.append((true_lat, true_lon, pred_lat, pred_lon))
    size_lat = min(LAT_MAX - LAT_MIN, 1.25 * max(
        np.ptp(np.concatenate([t, p.ravel()])) for t, _, p, _ in coords))
    size_lon = min(LON_MAX - LON_MIN, 1.25 * max(
        np.ptp(np.concatenate([t, p.ravel()])) for _, t, _, p in coords))

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=130)
    for ax, (rank, label), (true_lat, true_lon, pred_lat, pred_lon) in zip(
            axes.ravel(), picks, coords):
        track, preds_np, d = results[rank]

        add_basemap(ax)
        for i in range(CF.n_samples):
            ax.plot(pred_lon[i], pred_lat[i], color="#2b6cb0", alpha=0.45, linewidth=1.2,
                    zorder=2, label="predicted (16 samples)" if i == 0 else None)
        ax.plot(true_lon, true_lat, color="#c53030", linewidth=2.2, zorder=3, label="true path")
        ax.plot(true_lon[:init_seqlen], true_lat[:init_seqlen], "o", markersize=4.5,
                color="black", zorder=4, label=f"observed prompt ({init_seqlen} pts)")
        ax.plot(true_lon[init_seqlen - 1], true_lat[init_seqlen - 1], "*", markersize=16,
                color="#f6e05e", markeredgecolor="black", zorder=5, label="prediction start")

        zoom_window(ax, np.concatenate([true_lat, pred_lat.ravel()]),
                    np.concatenate([true_lon, pred_lon.ravel()]), size_lat, size_lon)

        errs = " | ".join(
            f"{h}h: {d[:, h * STEPS_PER_HOUR - 1].mean():.1f} km"
            for h in (1, 2, hours) if h * STEPS_PER_HOUR <= steps)
        ax.set_title(f"{label} — MMSI {track['mmsi']}\nmean error  {errs}", fontsize=10)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"TrAISformer {hours}-hour forecasts on held-out {CF.dataset_name} tracks "
        f"({CF.n_samples} sampled rollouts from an {init_seqlen}-point prompt)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dest = out_path("map_true_vs_pred.png")
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest}")


# ---------------------------------------------------------------------------
# 4. rollout probability heatmap animation
# ---------------------------------------------------------------------------

def make_rollout_heatmap_gif(steps=15):
    """animate the per-step lat x lon probability heatmap along one track

    Deliberately samples *without* the vicinity constraint (as eval_model.py does), so the frames
    show the raw distribution the model has learned rather than a window around the last position.
    """
    model = load_model()
    init_seqlen = CF.init_seqlen
    track = load_test_tracks(min_len=init_seqlen + steps + 1)[0]
    traj = track["traj"][:init_seqlen + steps]
    true_lat, true_lon = eval_model.undo_norm_ll(traj[:, 0], traj[:, 1])

    seq = torch.tensor(traj[np.newaxis, :init_seqlen, 0:4], dtype=torch.float32).to(CF.device)

    cmap = colormaps["jet"].copy()
    cmap.set_bad(alpha=0)

    tmpdir = tempfile.mkdtemp(prefix="rollout_")
    frame_paths = []
    for step in range(steps):
        with torch.no_grad():
            logits, _ = model(seq)
            lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
                logits[:, -1, :],
                (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        ix = torch.cat((torch.multinomial(lat_probs, num_samples=1),
                        torch.multinomial(lon_probs, num_samples=1),
                        torch.multinomial(sog_probs, num_samples=1),
                        torch.multinomial(cog_probs, num_samples=1)), dim=-1)
        d2inf_pred = torch.zeros((1, 4)).to(seq.device) + 0.5
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes
        seq = torch.cat((seq, x_sample.unsqueeze(1)), dim=1)

        # joint lat/lon probability over the ROI grid, normalized per frame. The distribution is
        # very peaked, so raise it to a fractional power before coloring or all but one bin
        # washes out to the bottom of the colormap.
        prob = np.outer(lat_probs.cpu().numpy().ravel(), lon_probs.cpu().numpy().ravel())
        prob = (prob / prob.max()) ** GAMMA
        prob[prob < 0.02] = np.nan

        pred_np = seq[0].cpu().numpy()
        pred_lat, pred_lon = eval_model.undo_norm_ll(pred_np[:, 0], pred_np[:, 1])

        fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
        add_basemap(ax)
        ax.plot(true_lon, true_lat, color="#c53030", linewidth=2, zorder=1, label="true path")
        ax.plot(true_lon[:init_seqlen], true_lat[:init_seqlen], ".", markersize=9,
                color="black", zorder=2, label="observed prompt")
        ax.plot(pred_lon[init_seqlen:], pred_lat[init_seqlen:], "x", markersize=8,
                color="#1a202c", zorder=4, label="sampled rollout")
        im = ax.imshow(prob, origin="lower", zorder=3, alpha=0.85,
                       extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), cmap=cmap,
                       vmin=0, vmax=1)
        # a square window that pans with the vessel, so the heatmap is big enough to read
        here_lat, here_lon = pred_lat[-1:], pred_lon[-1:]
        zoom_window(ax, here_lat, here_lon, WINDOW_DEG, WINDOW_DEG)
        fig.colorbar(im, ax=ax, shrink=0.8,
                     label=f"probability of the next position (scaled $p^{{{GAMMA}}}$)")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"step {step + 1}/{steps}  (+{(step + 1) * 10} min ahead)  — "
                     f"MMSI {track['mmsi']}", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()

        path = os.path.join(tmpdir, f"frame_{step:03d}.png")
        fig.savefig(path)
        plt.close(fig)
        frame_paths.append(path)

    save_gif(frame_paths, out_path("rollout_heatmap.gif"), duration=600, width=800)


if __name__ == "__main__":
    plot_loss_curves()
    make_epoch_gif()
    plot_map_true_vs_pred()
    make_rollout_heatmap_gif()
