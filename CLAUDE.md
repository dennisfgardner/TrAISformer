# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch implementation of TrAISformer (https://arxiv.org/abs/2109.03958), a GPT-style generative
transformer for AIS vessel-trajectory prediction. The transformer core is adapted from
karpathy/minGPT. This fork adds local visualization and evaluation scripts: `eval_model.py` (rolls
out a prediction and draws its probability heatmap over an OpenTopoMap basemap), `data_viewer.py`
(scatter-plots the raw dataset without a basemap), `make_visuals.py` (a finished run's figures) and
`compare_runs.py` (two runs on different regions, compared without letting geography do the talking).

## Environment & commands

Python 3.12 (`.python-version`), local `venv/` in the repo root, deps pinned in `requirements.txt`
(`requirements.yml` is the original upstream conda env and is stale).

```bash
source venv/bin/activate
python trAISformer.py       # train (if cf.retrain) + full test-set evaluation, writes prediction_error.png
python make_visuals.py      # loss curves, epoch animation, forecast maps, rollout heatmap for that run
python compare_runs.py measure   # cache this run's comparison numbers (once per dataset)
python compare_runs.py plot      # draw results/comparison/ from every cached measurement
python eval_model.py        # load checkpoint, roll out one test track, plot per-step lat/lon prob heatmap
python data_viewer.py       # scatter-plot raw test trajectories and coastline polygons
```

**These are slow, so plan before starting one.** On the RTX 3050, per epoch and per full test-set
pass:

| | tracks (train / test) | per epoch | test evaluation |
| --- | --- | --- | --- |
| `ct_dma` | 9,144 / 1,453 | ~2.4 min | ~12 min |
| `mc_ais` | 52,357 / 8,415 | ~13.5 min | ~70 min |

`trAISformer.py` does both, so a 15-epoch mc_ais run is about 4.5 hours. `compare_runs.py measure`
costs another full test pass. Run these in the background and ask the user about the epoch budget
before committing to one.

There is no test suite, linter config, or build step. `eval_model.py` / `data_viewer.py` use
`plt.show()` and need a GUI backend; `trAISformer.py` forces `matplotlib.use('Agg')`.

**If `import torch` fails** with `libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4`:
`~/.bashrc` puts `/usr/local/cuda-12.2/lib64` on `LD_LIBRARY_PATH`, so the system CUDA 12.2
`libnvJitLink.so.12` is found before the CUDA 12.4 copy that `torch==2.5.1+cu124` ships, and the
12.2 build lacks the symbol `libcusparse` needs. The venv works around it with
`site-packages/zz_nvjitlink_fix.pth`, which `ctypes.CDLL`s the bundled 12.4 library with
`RTLD_GLOBAL` at interpreter startup, so the right soname is already resolved when torch loads.
`venv/` is gitignored, so **recreate that `.pth` after rebuilding the venv** (or prepend
`$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib` to `LD_LIBRARY_PATH`).

## Configuration

`config_trAISformer.py` is the single source of truth — there is no CLI argument parsing. Every
script instantiates `Config()` and reads flags off it. Editing this file is how you change any
behavior.

**Values are overridden by re-assignment, not by editing in place.** The file leaves the previous
line intact and shadows it with a second assignment, so the *last* assignment wins:

```python
dataset_name = "ct_dma"
dataset_name = "mc_ais"                                  # <- the active dataset
...
datadir = f"./data/{dataset_name}/"
datadir = f"/home/dennis/repos/coast_guard_AIS_data/output/{dataset_name}/"   # <- the active path
```

To switch back, comment out or delete the second line; don't assume the first line reflects reality.

Other flags that matter:

- `retrain` — `False` skips training and only loads `cf.ckpt_path` for evaluation. Set `True` to train.
- `device` — hardcoded `cuda:0`; switch to the commented `cpu` line when no GPU is present.
- `savedir` / `ckpt_path` are **derived** from the hyperparameters via the long `filename` f-string,
  which begins with `dataset_name`. Changing the dataset — or any of `mode`, `sample_mode`, `top_k`,
  `r_vicinity`, the blur flags, the `*_size` bins, the `n_*_embd` sizes, `n_head`/`n_layer`,
  `batch_size`, `learning_rate`, or the seqlens — repoints `results/<filename>/model.pt` at a
  different (probably nonexistent) checkpoint. `results/` currently holds a `ct_dma-...-bs-16-...`
  and an `mc_ais-...-bs-16-...` run, both trained on this machine, plus a `.bak-2024-11-01` copy of
  the original upstream-default (`bs-32`) ct_dma checkpoint. `batch_size` is 16 rather than the
  upstream 32 because the 4 GB GPU runs out of memory at 32.

### State of the two trained runs

- **`ct_dma` is converged.** Validation loss bottomed at epoch 15 of 25 (1.386) and rose to 1.617 by
  epoch 25; `model.pt` is the epoch-15 checkpoint. More epochs would only overfit further.
- **`mc_ais` is not.** Validation loss fell monotonically to 2.042 at epoch 15, the last epoch, with
  a train/valid gap of 0.4 against ct_dma's 1.8 — with 5.7x the data it never got the chance to
  overfit, and the run stopped on the epoch budget rather than on a minimum. Treat every mc_ais
  number as a floor, and expect more epochs to improve it.
- The two loss values are **not comparable to each other**: each is a cross-entropy over its own
  region's grid.
- **Expect ~10% run-to-run variation** in the reported errors. The headline metric is the minimum
  over 16 sampled rollouts, the per-track errors are heavy-tailed, and nothing seeds the sampler per
  call, so `prediction_error.png` and `compare_runs.py` disagree by about that much on the *same*
  checkpoint. A difference at that scale is noise, not a regression.

### Regions of interest

Two ROI blocks are defined, guarded by `if dataset_name == ...`. Both use identical bin counts
(`lat_size=250`, `lon_size=270`, `sog_size=30`, `cog_size=72`) and embedding widths; only the
lat/lon bounds differ:

| `dataset_name` | ROI | lat | lon | Data location |
| --- | --- | --- | --- | --- |
| `ct_dma` | Danish straits | 55.5 – 58.0 | 10.3 – 13.0 | `data/ct_dma/` (in-repo) |
| `mc_ais` | Chesapeake Bay | 36.75 – 39.25 | -77.35 – -74.65 | `/home/dennis/repos/coast_guard_AIS_data/output/mc_ais/` (**outside the repo**) |

The `mc_ais` data comes from the separate `coast_guard_AIS_data` project (US Coast Guard / Marine
Cadastre 2023 AIS), whose `RegionOfInterest` dataclass defines the ROI this block must mirror; if
that directory is missing, `trAISformer.py` fails at the pickle load. Its bounds check is inclusive,
so a few positions land exactly on the edge and normalize to 1.0 rather than into `[0,1)` —
`TrAISformer.clamp_to_bins` keeps those from indexing past the end of an embedding table. The `mc_ais`
block is a second `if` rather than an `elif`, so both would execute if the names ever matched.

Both `eval_model.py` and `data_viewer.py` follow `Config`: each has a module-level `CF = Config()`
from which `LAT_MIN`/`LAT_MAX`/`LON_MIN`/`LON_MAX` and the test-set path (`CF.datadir` +
`CF.testset_name`) are derived. `eval_model.py` additionally keys its raster cache by dataset
(`basemap_<dataset_name>.tif`, gitignored as `basemap*.tif`), so switching `dataset_name` moves the
ROI, the data, the checkpoint, and the basemap together.

Because `Config` imports `torch`, both viewer scripts now pull in torch at import time.
`data_viewer.view_coastline_data` looks for `dma_coastline_polygons.pkl` under `CF.datadir` and prints
a skip message when absent — that file ships only with `ct_dma`, and no equivalent exists for `mc_ais`.


## Architecture

**Discretization is the central idea.** The four continuous channels (lat, lon, sog, cog) arrive
normalized to `[0,1)` and are binned into separate vocabularies (`lat_size=250`, `lon_size=270`,
`sog_size=30`, `cog_size=72`) by `TrAISformer.to_indexes`. Each channel gets its own `nn.Embedding`;
the four embeddings are **concatenated** (not summed) into `n_embd = 256+256+128+128 = 768`. The head
projects back to `full_size = 250+270+30+72 = 622` logits, which are `torch.split` into four
per-channel logit blocks and trained with four independent cross-entropy losses. Anything that
touches the head, the losses, or sampling must keep that split ordering `(lat, lon, sog, cog)`.

**Denormalization** back to degrees is `x * (MAX-MIN) + MIN` using the active ROI. The constants are
sourced from `Config` everywhere: `eval_model.py:undo_norm_ll`, `data_viewer.py:view_test_data`, and
the `v_ranges` / `v_roi_min` tensor pair `trAISformer.py` builds for the haversine error (now
`[model.lat_range, model.lon_range, 0, 0]` / `[model.lat_min, model.lon_min, 0, 0]`).

Upstream hardcoded that pair as `[2, 3, 0, 0]` / `[model.lat_min, -7, 0, 0]` — spans rounded off the
true ct_dma ROI (2.5 and 2.7) and a longitude origin that is simply wrong. The bogus longitude origin
was harmless, since `utils.haversine` uses longitude only through a difference and any offset
cancels; the rounded spans were not, and scaled the reported km. Fixing them raises ct_dma errors by
roughly 11%, so numbers from `prediction_error.png` no longer line up with the published paper
values even after the km-to-nautical-mile conversion.

**Blur loss** (`Config.blur*`): a fixed 1-D averaging `Conv1d` smooths the softmax over neighboring
bins and an extra NLL term on the blurred distribution is added to each channel loss, so
near-miss bins are not penalized as hard as distant ones.

**Vicinity-constrained sampling** (`trainers.sample`, `utils.top_k_nearest_idx`): with
`sample_mode="pos_vicinity"`, lat/lon logits farther than `r_vicinity/2` bins from the current
position are set to `-inf` before top-k and multinomial sampling. This is what keeps autoregressive
rollouts from teleporting. `eval_model.py` deliberately samples *without* this constraint so the raw
predicted distribution is visible in the heatmap.

### File map

- `models.py` — `CausalSelfAttention`, `Block`, `TrAISformer` (`forward` handles both the
  `with_targets` training path and the inference path), `configure_optimizers` (weight-decay
  whitelist/blacklist from minGPT).
- `trainers.py` — `sample()` (autoregressive rollout, used by both training viz and evaluation),
  `Trainer` (epoch loop, warmup+cosine LR schedule driven by a token counter, best-valid-loss
  checkpointing, per-epoch `epoch_XXX.jpg` trajectory plots into `savedir`).
- `datasets.py` — `AISDataset` (positions) and `AISDataset_grad` (positions + normalized deltas, used
  when `mode` is `"grad"`/`"pos_grad"`). Both pad/truncate to `max_seqlen` and return
  `(seq, mask, seqlen, mmsi, time_start)`.
- `utils.py` — seeding, log setup, `haversine` (returns km — the paper reports nautical miles),
  top-k helpers.
- `trAISformer.py` — the main entry point; also does the dataset filtering (drops tracks with NaNs,
  shorter than `min_seqlen`, and trims the leading not-yet-moving portion via `moving_threshold`).
- `make_visuals.py` — builds a finished run's figures into its `savedir`: `loss_curves.png` (parsed
  from the run log), `training_epochs.gif` (the trainer's per-epoch plots animated),
  `map_true_vs_pred.png` (forecast fan over the basemap, four tracks picked by error percentile) and
  `rollout_heatmap.gif`. `filter_tracks()` here mirrors trAISformer.py's filtering.
- `compare_runs.py` — compares two runs on different regions. `measure` scores the active dataset's
  whole test set against a dead-reckoning baseline and caches `comparison_metrics.json` beside the
  checkpoint; `plot` reads every cached measurement and writes `results/comparison/`. Run `measure`
  once per dataset, since `Config` only describes one at a time.

Note `Config.mode` advertises many variants (`"velo"`, `"grid_l2"`, `"gridcont_*"`, …) that are
referenced in branches but only `"pos"` is exercised; `models.py` also has `mode`-specific branches
whose supporting code (e.g. `res_pred`) is absent.

## Data

Pickled lists of dicts with `"mmsi"` and `"traj"`, where `traj` columns are
`[lat, lon, sog, cog, unix_timestamp, mmsi]` and lat/lon/sog/cog are already normalized to `[0,1)`
against that dataset's ROI. Split into `<name>_train.pkl` / `_valid.pkl` / `_test.pkl`.

- `data/ct_dma/` — Danish Maritime Authority data shipped with the upstream repo; preprocessing code
  lives in the GeoTrackNet repo (see README). `dma_coastline_polygons.pkl` holds un-normalized
  `(lat, lon)` coastline arrays, used only by `data_viewer.py`.
- `/home/dennis/repos/coast_guard_AIS_data/output/mc_ais/` — US Coast Guard / Marine Cadastre AIS
  data for Chesapeake Bay, generated outside this repo.

`basemap_<dataset_name>.tif` is a cached contextily/OpenTopoMap raster covering that dataset's ROI,
written on first use by `eval_model.save_basemap()` and gitignored via `basemap*.tif`. Delete one to
force a refetch. The ct_dma raster was originally named `basemap.tif`; it was renamed to
`basemap_ct_dma.tif` when the cache became per-dataset.
