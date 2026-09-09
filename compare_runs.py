#!/usr/bin/env python3

"""Compare two trained runs on different regions, without letting geography do the talking.

Two datasets cover different water, so raw kilometres are not on their own a fair comparison: a
region whose traffic runs in straight lanes is easier than one full of junctions, whatever the model
is worth. This script measures each run against its own region and reports the cuts that survive
that difference:

    skill vs dead reckoning   error relative to a constant-velocity extrapolation of the prompt
    relative error            error as a fraction of how far the vessel actually travelled
    error distribution        the whole per-track CDF, not just its mean
    learning curves           validation loss against gradient steps, not epochs

`Config` describes one dataset at a time, so measurement runs once per dataset against whichever one
is active, caching its numbers next to that run's checkpoint:

    python compare_runs.py measure     # with config_trAISformer.py pointed at dataset A
    python compare_runs.py measure     # ... and again with it pointed at dataset B
    python compare_runs.py plot        # reads every cached measurement, writes results/comparison/
"""

import os
import re
import sys
import json
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

import trainers  # noqa: F401 -- also forces the Agg backend via trAISformer
import utils
import make_visuals as mv
from config_trAISformer import Config

CF = Config()

# one AIS report every 10 minutes
STEPS_PER_HOUR = mv.STEPS_PER_HOUR
HORIZON_HOURS = 4
HORIZONS = (1, 2, 3, 4)

# dead reckoning fits a velocity to the last few prompt steps rather than just the last one, so a
# single noisy report does not set the whole extrapolation
DR_FIT_STEPS = 3

# optional cap on test tracks measured; None uses the whole test set, which is what keeps two
# datasets of different sizes on the same footing
MAX_TRACKS = None

COMPARE_DIR = os.path.join("results", "comparison")
METRICS_NAME = "comparison_metrics.json"


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def train_set_size():
    """number of training tracks this run used, i.e. the train set after the same filtering

    trAISformer.py only prints this to stdout, so it is recomputed here rather than parsed.
    """
    import pickle
    with open(os.path.join(CF.datadir, CF.trainset_name), "rb") as f:
        return len(mv.filter_tracks(pickle.load(f), min_len=CF.min_seqlen + 1))


def to_degrees(seqs, model):
    """(lat, lon) in degrees from normalized sequences of shape (..., >=2)"""
    lat = seqs[..., 0] * model.lat_range + model.lat_min
    lon = seqs[..., 1] * model.lon_range + model.lon_min
    return lat, lon


def haversine_deg(lat_a, lon_a, lat_b, lon_b):
    """great-circle distance in km between two (..., T) sets of degree coordinates"""
    a = torch.stack([lat_a, lon_a], dim=-1) * np.pi / 180
    b = torch.stack([lat_b, lon_b], dim=-1) * np.pi / 180
    return utils.haversine(a, b)


def dead_reckon(truth, model, init_seqlen, steps):
    """constant-velocity extrapolation of the prompt, in degrees

    Fits one displacement per step from the last DR_FIT_STEPS intervals of the observed prompt and
    keeps applying it. This is what a navigator does with a ruler, and it is the bar any forecast
    model has to clear.
    """
    lat, lon = to_degrees(truth, model)
    last = init_seqlen - 1
    d_lat = (lat[:, last] - lat[:, last - DR_FIT_STEPS]) / DR_FIT_STEPS
    d_lon = (lon[:, last] - lon[:, last - DR_FIT_STEPS]) / DR_FIT_STEPS
    k = torch.arange(1, steps + 1, device=truth.device).float()
    return (lat[:, last:last + 1] + d_lat[:, None] * k,
            lon[:, last:last + 1] + d_lon[:, None] * k)


def measure():
    """roll the active run out over its test set and cache the per-track numbers"""
    import datasets
    import pickle

    model = mv.load_model()
    init_seqlen = CF.init_seqlen
    steps = HORIZON_HOURS * STEPS_PER_HOUR

    with open(os.path.join(CF.datadir, CF.testset_name), "rb") as f:
        raw = pickle.load(f)
    tracks = mv.filter_tracks(raw, min_len=CF.min_seqlen + 1)
    if MAX_TRACKS is not None:
        rng = np.random.default_rng(42)
        tracks = [tracks[i] for i in rng.permutation(len(tracks))[:MAX_TRACKS]]
    dataset = datasets.AISDataset(tracks, max_seqlen=CF.max_seqlen + 1, device=CF.device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CF.batch_size, shuffle=False)
    print(f"{CF.dataset_name}: measuring {len(tracks)} test tracks, {HORIZON_HOURS}h horizon")

    min_err, mean_err, dr_err, travelled, valid = [], [], [], [], []
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in enumerate(loader):
            seqs = seqs[:, :init_seqlen + steps, :].to(CF.device).float()
            masks = masks[:, :init_seqlen + steps].to(CF.device)
            batch = seqs.shape[0]

            true_lat, true_lon = to_degrees(seqs, model)
            fut_lat = true_lat[:, init_seqlen:]
            fut_lon = true_lon[:, init_seqlen:]

            ens = torch.zeros((batch, steps, CF.n_samples), device=CF.device)
            for i_sample in range(CF.n_samples):
                preds = trainers.sample(model, seqs[:, :init_seqlen, :], steps,
                                        temperature=1.0, sample=True,
                                        sample_mode=CF.sample_mode,
                                        r_vicinity=CF.r_vicinity,
                                        top_k=CF.top_k)
                p_lat, p_lon = to_degrees(preds[:, init_seqlen:], model)
                ens[:, :, i_sample] = haversine_deg(fut_lat, fut_lon, p_lat, p_lon)

            dr_lat, dr_lon = dead_reckon(seqs, model, init_seqlen, steps)

            min_err.append(ens.min(dim=-1).values.cpu().numpy())
            mean_err.append(ens.mean(dim=-1).cpu().numpy())
            dr_err.append(haversine_deg(fut_lat, fut_lon, dr_lat, dr_lon).cpu().numpy())
            # distance actually covered from the prediction start, the natural scale for the error
            travelled.append(haversine_deg(
                true_lat[:, init_seqlen - 1:init_seqlen].expand(-1, steps),
                true_lon[:, init_seqlen - 1:init_seqlen].expand(-1, steps),
                fut_lat, fut_lon).cpu().numpy())
            valid.append(masks[:, init_seqlen:].cpu().numpy().astype(bool))

            if it % 25 == 0:
                print(f"  batch {it}/{len(loader)}", flush=True)

    metrics = {
        "dataset_name": CF.dataset_name,
        "roi": [CF.lat_min, CF.lat_max, CF.lon_min, CF.lon_max],
        "n_test_tracks": len(tracks),
        "n_train_tracks": train_set_size(),
        "batch_size": CF.batch_size,
        "init_seqlen": init_seqlen,
        "steps": steps,
        "savedir": CF.savedir,
        "losses": read_losses(),
        "min_err": np.concatenate(min_err).tolist(),
        "mean_err": np.concatenate(mean_err).tolist(),
        "dr_err": np.concatenate(dr_err).tolist(),
        "travelled": np.concatenate(travelled).tolist(),
        "valid": np.concatenate(valid).tolist(),
    }
    dest = os.path.join(CF.savedir, METRICS_NAME)
    with open(dest, "w") as f:
        json.dump(metrics, f)
    print(f"wrote {dest}")
    summarize(Run(metrics))


def read_losses():
    """train/valid loss per epoch from the run's log"""
    text = open(mv.newest_log()).read()
    train = {int(e): float(v) for e, v in
             re.findall(r"Training, epoch (\d+), loss (-?[\d.]+),", text)}
    valid = {int(e): float(v) for e, v in
             re.findall(r"Valid, epoch (\d+), loss (-?[\d.]+)\.", text)}
    return {"train": train, "valid": valid}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

class Run:
    """one measured run, with the derived curves the comparison plots need"""

    def __init__(self, metrics):
        self.__dict__.update({k: v for k, v in metrics.items()
                              if k not in ("min_err", "mean_err", "dr_err", "travelled", "valid")})
        self.min_err = np.array(metrics["min_err"])
        self.mean_err = np.array(metrics["mean_err"])
        self.dr_err = np.array(metrics["dr_err"])
        self.travelled = np.array(metrics["travelled"])
        self.valid = np.array(metrics["valid"], dtype=bool)
        self.hours = (np.arange(self.steps) + 1) / STEPS_PER_HOUR

    def curve(self, arr):
        """mean over tracks at each step, ignoring padded steps"""
        counts = self.valid.sum(axis=0)
        return np.where(counts > 0, (arr * self.valid).sum(axis=0) / np.maximum(counts, 1), np.nan)

    def at(self, arr, hour):
        """per-track values at a horizon, for tracks that actually reach it"""
        step = hour * STEPS_PER_HOUR - 1
        return arr[self.valid[:, step], step]

    def skill(self, hour):
        """1 - model error / dead reckoning error, paired per track"""
        step = hour * STEPS_PER_HOUR - 1
        keep = self.valid[:, step]
        return 1 - self.min_err[keep, step].mean() / self.dr_err[keep, step].mean()

    def relative(self, hour):
        """error as a fraction of the distance covered, as a ratio of means

        A mean of per-track ratios is useless here: a vessel that barely moved contributes a
        division by almost zero and swamps everything else.
        """
        return self.at(self.min_err, hour).mean() / self.at(self.travelled, hour).mean()

    @property
    def steps_per_epoch(self):
        return int(np.ceil(self.n_train_tracks / self.batch_size))

    def loss_by_step(self, which):
        losses = self.losses[which]
        epochs = sorted(int(e) for e in losses)
        return ([e * self.steps_per_epoch for e in epochs],
                [losses[str(e)] if str(e) in losses else losses[e] for e in epochs])


def load_runs():
    """every cached measurement under results/"""
    runs = []
    for path in sorted(glob.glob(os.path.join("results", "*", METRICS_NAME))):
        with open(path) as f:
            runs.append(Run(json.load(f)))
    if len(runs) < 2:
        raise SystemExit(f"found {len(runs)} measurement(s); run `measure` for each dataset first")
    return runs


def summarize(run):
    """print the headline numbers for one run"""
    print(f"\n{run.dataset_name}: {run.n_test_tracks} test tracks")
    print(f"{'horizon':>8} {'best-of-16':>11} {'mean-of-16':>11} {'dead reck.':>11} "
          f"{'skill':>7} {'err/dist':>9}")
    for h in HORIZONS:
        rel = run.relative(h)
        print(f"{h:>7}h {run.at(run.min_err, h).mean():>10.2f}k {run.at(run.mean_err, h).mean():>10.2f}k "
              f"{run.at(run.dr_err, h).mean():>10.2f}k {run.skill(h):>7.2f} {rel:>9.1%}")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

COLORS = {"ct_dma": "#1d6f8b", "mc_ais": "#ab3b26"}
LABELS = {"ct_dma": "ct_dma — Danish straits", "mc_ais": "mc_ais — Chesapeake Bay"}


def color(run):
    return COLORS.get(run.dataset_name, "#555555")


def label(run):
    return LABELS.get(run.dataset_name, run.dataset_name)


def plot_skill(runs, dest):
    """error against the dead reckoning baseline, and the skill score it implies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    for run in runs:
        c = color(run)
        ax1.plot(run.hours, run.curve(run.min_err), "-", color=c, linewidth=2,
                 label=f"{label(run)} — model")
        ax1.plot(run.hours, run.curve(run.dr_err), "--", color=c, linewidth=1.4, alpha=.75,
                 label=f"{label(run)} — dead reckoning")
        ax2.plot(HORIZONS, [run.skill(h) for h in HORIZONS], "-o", color=c, linewidth=2,
                 markersize=5, label=label(run))

    ax1.set_xlabel("forecast horizon (hours)")
    ax1.set_ylabel("mean error (km)")
    ax1.set_title("Model vs constant-velocity extrapolation")
    ax1.grid(alpha=.25)
    ax1.legend(fontsize=8.5)

    ax2.axhline(0, color="0.5", linewidth=1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(HORIZONS)
    ax2.set_xlabel("forecast horizon (hours)")
    ax2.set_ylabel("skill score  (1 − model / dead reckoning)")
    ax2.set_title("Skill: fraction of the baseline's error removed")
    ax2.grid(alpha=.25)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest}")


def plot_cdf(runs, dest):
    """the whole distribution of per-track error, at each horizon"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150, sharey=True)
    for ax, hour in zip(axes, (1, 2, 3)):
        for run in runs:
            e = np.sort(run.at(run.min_err, hour))
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=color(run), linewidth=2,
                    label=label(run))
            median = np.median(e)
            ax.plot([median], [0.5], "o", color=color(run), markersize=5)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 200)
        ax.axhline(0.5, color="0.6", linewidth=.9, linestyle=":")
        ax.axhline(0.9, color="0.6", linewidth=.9, linestyle=":")
        ax.set_xlabel("best-of-16 error at %dh (km, log scale)" % hour)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("fraction of tracks at or below")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Error distribution — dotted lines mark the median and the 90th percentile",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest}")


def plot_relative(runs, dest):
    """error as a fraction of the distance the vessel actually covered"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    width = 0.36
    for i, run in enumerate(runs):
        c = color(run)
        ax1.plot(run.hours, run.curve(run.travelled), "-", color=c, linewidth=2, label=label(run))
        rel = [run.relative(h) * 100 for h in HORIZONS]
        ax2.bar(np.array(HORIZONS) + (i - (len(runs) - 1) / 2) * width, rel, width,
                color=c, label=label(run))
        for h, v in zip(HORIZONS, rel):
            ax2.text(h + (i - (len(runs) - 1) / 2) * width, v + .4, f"{v:.1f}%",
                     ha="center", fontsize=8.5, color=c)

    ax1.set_xlabel("forecast horizon (hours)")
    ax1.set_ylabel("distance travelled from prediction start (km)")
    ax1.set_title("How far these fleets actually move")
    ax1.grid(alpha=.25)
    ax1.legend(fontsize=9)

    ax2.set_xticks(HORIZONS)
    ax2.set_xlabel("forecast horizon (hours)")
    ax2.set_ylabel("error as % of distance travelled")
    ax2.set_title("Error relative to the distance being predicted")
    ax2.grid(alpha=.25, axis="y")

    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest}")


def plot_learning(runs, dest):
    """validation loss against gradient steps, so dataset size is visible"""
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150)
    left, right = np.inf, 0
    for run in runs:
        c = color(run)
        for which, style, alpha in (("train", "--", .6), ("valid", "-", 1.0)):
            x, y = run.loss_by_step(which)
            ax.plot(x, y, style, color=c, linewidth=2 if which == "valid" else 1.3, alpha=alpha,
                    label=f"{label(run)} — {which}")
        x, y = run.loss_by_step("valid")
        best = int(np.argmin(y))
        note = "still falling" if best == len(y) - 1 else "turned here"
        ax.plot([x[best]], [y[best]], "o", color=c, markersize=7, zorder=5)
        ax.annotate(f"best: epoch {best + 1}\n{y[best]:.3f} — {note}", xy=(x[best], y[best]),
                    xytext=(10, 16), textcoords="offset points", fontsize=8.5, color=c)
        left = min(left, x[0])
        right = max(right, x[-1])

    # log scale first: an xlim set before the scale change is reinterpreted by it. The last
    # annotation sits to the right of the last point, so leave it somewhere to go.
    ax.set_xscale("log")
    ax.set_xlim(left * 0.8, right * 2.4)
    ax.set_xlabel("gradient steps (log scale)")
    ax.set_ylabel("loss")
    ax.set_title("Learning curves on a common axis — steps, not epochs")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest}")


def plot():
    runs = load_runs()
    os.makedirs(COMPARE_DIR, exist_ok=True)
    for run in runs:
        summarize(run)
    plot_skill(runs, os.path.join(COMPARE_DIR, "skill_vs_dead_reckoning.png"))
    plot_cdf(runs, os.path.join(COMPARE_DIR, "error_distribution.png"))
    plot_relative(runs, os.path.join(COMPARE_DIR, "relative_error.png"))
    plot_learning(runs, os.path.join(COMPARE_DIR, "learning_curves.png"))
    with open(os.path.join(COMPARE_DIR, "summary.json"), "w") as f:
        json.dump({run.dataset_name: {
            "n_test_tracks": run.n_test_tracks,
            "n_train_tracks": run.n_train_tracks,
            "steps_per_epoch": run.steps_per_epoch,
            "roi": run.roi,
            **{f"{h}h": {
                "min_err_km": float(run.at(run.min_err, h).mean()),
                "mean_err_km": float(run.at(run.mean_err, h).mean()),
                "dr_err_km": float(run.at(run.dr_err, h).mean()),
                "median_err_km": float(np.median(run.at(run.min_err, h))),
                "p90_err_km": float(np.percentile(run.at(run.min_err, h), 90)),
                "skill": float(run.skill(h)),
                "travelled_km": float(run.at(run.travelled, h).mean()),
                "rel_err": float(run.relative(h)),
            } for h in HORIZONS},
        } for run in runs}, f, indent=2)
    print(f"wrote {os.path.join(COMPARE_DIR, 'summary.json')}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if action == "measure":
        if len(sys.argv) > 2:
            MAX_TRACKS = int(sys.argv[2])
        measure()
    elif action == "plot":
        plot()
    else:
        raise SystemExit(__doc__)
