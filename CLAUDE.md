# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch implementation of TrAISformer (https://arxiv.org/abs/2109.03958), a GPT-style generative
transformer for AIS vessel-trajectory prediction. The transformer core is adapted from
karpathy/minGPT. This fork adds local evaluation/visualization scripts (`eval_model.py`,
`data_viewer.py`) that plot predictions over an OpenTopoMap basemap.

## Environment & commands

Python 3.12 (`.python-version`), local `venv/` in the repo root, deps pinned in `requirements.txt`
(`requirements.yml` is the original upstream conda env and is stale).

```bash
source venv/bin/activate
python trAISformer.py     # train (if cf.retrain) + full test-set evaluation, writes prediction_error.png
python eval_model.py      # load checkpoint, roll out one test track, plot per-step lat/lon prob heatmap
python data_viewer.py     # scatter-plot raw test trajectories and coastline polygons
```

There is no test suite, linter config, or build step. `eval_model.py` / `data_viewer.py` use
`plt.show()` and need a GUI backend; `trAISformer.py` forces `matplotlib.use('Agg')`.

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
datadir = f"../marine-cadastre/output/{dataset_name}/"   # <- the active path
```

To switch back, comment out or delete the second line; don't assume the first line reflects reality.

Other flags that matter:

- `retrain` — `False` skips training and only loads `cf.ckpt_path` for evaluation. Set `True` to train.
- `device` — hardcoded `cuda:0`; switch to the commented `cpu` line when no GPU is present.
- `savedir` / `ckpt_path` are **derived** from the hyperparameters via the long `filename` f-string,
  which begins with `dataset_name`. Changing the dataset — or any of `mode`, `sample_mode`, `top_k`,
  `r_vicinity`, the blur flags, the `*_size` bins, the `n_*_embd` sizes, `n_head`/`n_layer`,
  `batch_size`, `learning_rate`, or the seqlens — repoints `results/<filename>/model.pt` at a
  different (probably nonexistent) checkpoint. The only trained checkpoint in `results/` is the
  `ct_dma-...` one matching the upstream defaults, so loading it requires `dataset_name = "ct_dma"`.

### Regions of interest

Two ROI blocks are defined, guarded by `if dataset_name == ...`. Both use identical bin counts
(`lat_size=250`, `lon_size=270`, `sog_size=30`, `cog_size=72`) and embedding widths; only the
lat/lon bounds differ:

| `dataset_name` | ROI | lat | lon | Data location |
| --- | --- | --- | --- | --- |
| `ct_dma` | Danish straits | 55.5 – 58.0 | 10.3 – 13.0 | `data/ct_dma/` (in-repo) |
| `mc_ais` | US Mid-Atlantic | 36.75 – 39.25 | -77.35 – -74.65 | `../marine-cadastre/output/mc_ais/` (**outside the repo**) |

The `mc_ais` data comes from a separate sibling `marine-cadastre` project that is not part of this
repository; if that directory is missing, `trAISformer.py` fails at the pickle load. The `mc_ais`
block is a second `if` rather than an `elif`, so both would execute if the names ever matched.

`eval_model.py` follows `Config`: its module-level `CF = Config()` derives `LAT_MIN`/`LAT_MAX`/
`LON_MIN`/`LON_MAX`, the test-set path (`CF.datadir` + `CF.testset_name`), and a per-dataset raster
cache `basemap_<dataset_name>.tif` (gitignored as `basemap*.tif`), so switching `dataset_name` moves
the ROI, the data, the checkpoint, and the basemap together.

**`data_viewer.py` still does not honor `dataset_name`** — it hardcodes the `ct_dma` ROI constants
inside `view_test_data` and the literal paths `data/ct_dma/ct_dma_test.pkl` and
`data/ct_dma/dma_coastline_polygons.pkl`, so it always plots the Danish dataset.


## Architecture

**Discretization is the central idea.** The four continuous channels (lat, lon, sog, cog) arrive
normalized to `[0,1)` and are binned into separate vocabularies (`lat_size=250`, `lon_size=270`,
`sog_size=30`, `cog_size=72`) by `TrAISformer.to_indexes`. Each channel gets its own `nn.Embedding`;
the four embeddings are **concatenated** (not summed) into `n_embd = 256+256+128+128 = 768`. The head
projects back to `full_size = 250+270+30+72 = 622` logits, which are `torch.split` into four
per-channel logit blocks and trained with four independent cross-entropy losses. Anything that
touches the head, the losses, or sampling must keep that split ordering `(lat, lon, sog, cog)`.

**Denormalization** back to degrees is `x * (MAX-MIN) + MIN` using the active ROI. The constants are
sourced from `Config` in `eval_model.py:undo_norm_ll`, but still hardcoded to `ct_dma` in
`data_viewer.py:view_test_data`. `trAISformer.py` takes a third route for the haversine error — a
`v_ranges` / `v_roi_min` tensor pair fixed at `[2, 3, 0, 0]` / `[model.lat_min, -7, 0, 0]`, which
encodes the Danish lat/lon spans and is therefore wrong for any other ROI.

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
- `../marine-cadastre/output/mc_ais/` — US Marine Cadastre AIS data, generated outside this repo.

`basemap.tif` is a cached contextily/OpenTopoMap raster for the `ct_dma` ROI, gitignored. Delete it
to have `eval_model.save_basemap()` refetch — necessary if the ROI changes, since the filename is fixed.
