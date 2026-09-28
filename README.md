# CM3070 Final Project - Neural Style Transfer: Architecture, Hyperparameters and Originality

**Learning and Neural Networks, Project Idea 1: Neural Style Transfer**.

What is covered:

* optimisation-based NST (Gatys et al., 2016) with **VGG-16 / VGG-19**, **max / average pooling**,
  configurable content and style layers, and Gatys-style per-channel feature normalisation
* three optimisers: **L-BFGS**, **Adam** and **SGD with momentum**
* content, noise or style **initialisation**
* **colour preservation** (luminance-only transfer, Gatys et al., 2017)
* **multi-style blending** (convex combination of Gram targets)
* **video NST** with warm-starting and a temporal consistency loss (after Ruder et al., 2016)
* an **evaluation module** with network-independent metrics (SSIM, edge F1, CIELAB colour
  distance, LBP texture distance) plus a fixed deep evaluator network
* an experiment runner that logs every run's configuration, loss curve, runtime and metrics

## Layout
```
nst/                  core package
  models.py           VGG-16/19 builder, weight loading, feature normalisation
  losses.py           Gram matrix, content/style/TV losses, multi-style targets
  core.py             NSTConfig + stylise() optimisation loop
  colour.py           YIQ conversion for colour preservation
  metrics.py          evaluation metrics
  io.py               image loading/saving, figure grids
run_nst.py            command-line interface
experiments/
  run_experiments.py  experiments E1-E9 (results/runs.csv, results/images, results/logs)
  video_nst.py        video NST experiment (results/video)
  make_figures.py     all report figures and tables
tests/test_nst.py     unit tests (pytest)
data/                 content and style images + SOURCES.md (provenance/licences)
weights/              pretrained VGG weights (see below)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p weights
curl -L -o weights/vgg19_notop.h5 https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
curl -L -o weights/vgg16_notop.h5 https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
```
These are the ImageNet VGG weights used by Keras and by Chollet (2018). To use torchvision's
weights instead, pass `--weights torchvision` (downloaded automatically on first use).

## Usage
```bash
python run_nst.py --content data/content/tubingen.jpg --style data/style/starry_night.jpg \
    --out outputs/tubingen_starry.png --style-weight 1e5 --optimizer lbfgs --steps 300

# colour preservation, VGG-16 with average pooling
python run_nst.py --content data/content/me.jpg --style data/style/the_scream.jpg \
    --preserve-colour --arch vgg16 --pooling avg --out outputs/portrait.png

# blend two styles
python run_nst.py --content data/content/tubingen.jpg \
    --style data/style/starry_night.jpg data/style/the_scream.jpg --style-mix 0.5 0.5
```

Reproduce the report:
```bash
python experiments/run_experiments.py          
python experiments/video_nst.py
python experiments/make_figures.py
python -m pytest tests -q
```

Image provenance is documented in `data/SOURCES.md`.
