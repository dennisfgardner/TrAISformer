# TrAISformer

Pytorch implementation of TrAISformer---A generative transformer for AIS trajectory prediction (https://arxiv.org/abs/2109.03958).

The transformer part is adapted from: https://github.com/karpathy/minGPT

---
<p align="center">
  <img width="600" height="450" src="./figures/t18_3.png">
</p>

---

### About this fork

Fork of [CIA-Oceanix/TrAISformer](https://github.com/CIA-Oceanix/TrAISformer), updated to run on
Ubuntu with Python 3.12 and extended with visualization and a second dataset. Changes relative to
upstream:

- **Runs on Python 3.12 / PyTorch 2.5** — `iter(dl).next()` replaced with `next(iter(dl))`, and the
  timestamp cast in `datasets.py` made explicit.
- **Two regions of interest**, selected by `dataset_name` in `config_trAISformer.py`: the original
  Danish `ct_dma` and `mc_ais`, US Mid-Atlantic data built from
  [Marine Cadastre](https://marinecadastre.gov/ais/) AIS records by a separate sibling project.
- **The ROI is read from `Config` everywhere.** `eval_model.py`, `data_viewer.py` and the haversine
  error in `trAISformer.py` previously hardcoded the Danish bounds, so they silently produced wrong
  results for any other region. See the note under [Run](#run) — this changes the reported errors.
- **New `eval_model.py`** — loads a checkpoint, rolls a test track forward step by step, and plots
  the predicted lat/lon probability distribution as a heatmap over an OpenTopoMap basemap.
- **New `data_viewer.py`** — scatter-plots the raw test trajectories and the coastline polygons.
- **`CLAUDE.md`** — architecture and configuration notes for working in this repository.

### Requirements

Python 3.12 with the pinned versions in `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.yml` is upstream's original conda environment and is no longer maintained here.

> **If `import torch` fails** with `libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4`,
> a system CUDA install earlier on `LD_LIBRARY_PATH` is shadowing the CUDA 12.4 `libnvJitLink.so.12`
> that `torch==2.5.1+cu124` bundles. Put the venv's own copy first:
> ```bash
> export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
> ```

### Datasets:

The data used in this paper are provided by the [Danish Maritime Authority (DMA)](https://dma.dk/safety-at-sea/navigational-information/ais-data). 
Please refer to [the paper](https://arxiv.org/abs/2109.03958) for the details of the pre-processing step. The code is available here: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py

A processed dataset can be found in `./data/ct_dma/`
(the format is `[lat, log, sog, cog, unix_timestamp, mmsi]`).

### Run

All settings live in `config_trAISformer.py`; there are no command-line arguments. Set `retrain =
True` to train, `False` to load an existing checkpoint and evaluate only. Note that `savedir` and
`ckpt_path` are derived from the hyperparameters, so changing any of them points at a different
checkpoint directory.

```bash
python trAISformer.py     # train (if retrain) and evaluate, writes prediction_error.png
python eval_model.py      # roll out one test track, plot the prediction heatmap on a basemap
python data_viewer.py     # scatter-plot the raw trajectories and coastline
```

Prediction errors are reported in km, while the paper converts them to nautical miles.

> **Errors differ from the published values.** Upstream denormalized the coordinates for the
> haversine distance with the ROI spans rounded to `[2, 3]`; the true `ct_dma` spans are `2.5` and
> `2.7`. This fork uses the actual ROI from `Config`, which raises the reported `ct_dma` errors by
> roughly 11%. The new numbers are the geometrically correct ones — reproducing the paper's table
> exactly requires the old rounded constants.


### License

See `LICENSE`

### Contact
For any questions, please open an issue and assign it to @dnguyengithub.

