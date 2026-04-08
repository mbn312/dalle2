# DALL-E 2 From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g4kWCBPaNEq1YBeBB7iwKeU3_3LRcEFf?usp=sharing)

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/correll-lab/dall-e-2-from-scratch-c055bf881b9a)

This repository is a compact PyTorch implementation of a DALL-E 2 style `unCLIP` pipeline trained on Fashion-MNIST. It is not a reproduction of OpenAI's production-scale DALL-E 2 system. Instead, it is a small, readable project that trains three stages end to end:

1. A CLIP-like image/text embedding model
2. A diffusion prior that maps text embeddings to CLIP image embeddings
3. A diffusion decoder that turns those image embeddings into `32x32` grayscale images

The codebase ships with pretrained checkpoints in [`trained_models/`](./trained_models), a notebook in [`notebooks/diffusion_samples.ipynb`](./notebooks/diffusion_samples.ipynb), and standalone training scripts for each stage.

More detailed implementation notes live in [`docs/architecture.md`](./docs/architecture.md).

## Project Scope

This project is intentionally narrow:

- Dataset: Fashion-MNIST only
- Images: grayscale, resized from `28x28` to `32x32`
- Text: fixed class captions encoded with a simple byte-level tokenizer
- Output space: clothing-like Fashion-MNIST samples, not open-domain image generation

That scope is consistent with the accompanying article, which describes the project as "Text-Conditioned Image Generation on FashionMNIST using CLIP Latents."

## Architecture At A Glance

The end-to-end generation path is:

`caption -> CLIP text encoder -> diffusion prior -> CLIP image embedding -> diffusion decoder -> image`

At the implementation level:

- [`model/clip.py`](./model/clip.py) defines a Vision Transformer image encoder, a Transformer text encoder, and the symmetric contrastive CLIP loss.
- [`model/prior.py`](./model/prior.py) freezes the trained CLIP model and learns to predict CLIP image embeddings from caption information plus diffusion timestep conditioning.
- [`model/decoder.py`](./model/decoder.py) freezes the prior and uses a UNet-style diffusion model with residual and attention blocks to predict image noise.
- [`data/dataset.py`](./data/dataset.py) wraps Fashion-MNIST and maps labels to fixed natural-language captions.
- [`data/FMNISTConfig.py`](./data/FMNISTConfig.py) holds all model and training hyperparameters.

## Setup

If you want to use the shipped checkpoints in [`trained_models/`](./trained_models), install Git LFS first. Without Git LFS, those `.pt` files may remain pointer files, `isfile()` checks will still pass, and `torch.load()` will fail later.

Create a Python environment, pull the LFS assets, and then install the dependencies:

```bash
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

`requirements.txt` pins CUDA 12.1 PyTorch wheels:

- `torch==2.2.0+cu121`
- `torchvision==0.17.0+cu121`

If you are on CPU-only hardware or a different CUDA version, do not use the pinned `requirements.txt` as-is. Install compatible `torch` and `torchvision` packages for your environment first, then install the remaining Python dependencies separately.

By default the dataset downloads to a `datasets/` directory next to the repository root. You can change that in [`data/FMNISTConfig.py`](./data/FMNISTConfig.py).

## Quick Start

The fastest way to inspect the model behavior is the notebook:

- Local notebook: [`notebooks/diffusion_samples.ipynb`](./notebooks/diffusion_samples.ipynb)
- Colab version: <https://colab.research.google.com/drive/1g4kWCBPaNEq1YBeBB7iwKeU3_3LRcEFf?usp=sharing>

The notebook will train any missing prerequisite checkpoints and then plot intermediate reverse-diffusion frames for each Fashion-MNIST caption.

You can also sample programmatically from the repository root:

```python
import torch
import matplotlib.pyplot as plt
from data.FMNISTConfig import FMNISTConfig
from data.data_utils import tokenizer
from model.decoder import sample_image

config = FMNISTConfig()
caption = "An image of a sneaker"

tokens, mask = tokenizer(caption, text_seq_length=config.text_seq_length)
tokens = tokens[None].to(config.device)
mask = mask[None].to(config.device)

image = sample_image(config, tokens, mask)

plt.imshow(image.detach().cpu()[0].permute(1, 2, 0), cmap="gray")
plt.axis("off")
plt.show()
```

This loads the decoder checkpoint, which in turn loads the prior and CLIP checkpoints automatically. The plotting helper notebook still uses `sample_plot_image()`, but the lower-level `sample_image()` path is the safest programmatic entry point on both CPU and CUDA.

## Training

Each stage has its own entrypoint:

```bash
python3 train_clip.py
python3 train_prior.py
python3 train_decoder.py
```

Dependency order matters:

- `train_clip.py` trains only the CLIP stage.
- `train_prior.py` requires a CLIP checkpoint and will train CLIP first if it is missing.
- `train_decoder.py` requires both the CLIP and prior checkpoints and will train missing dependencies first.

Default hyperparameters are defined in [`data/FMNISTConfig.py`](./data/FMNISTConfig.py). The shipped defaults are:

- CLIP: `200` epochs, batch size `128`
- Prior: `150` epochs, batch size `128`
- Decoder: `100` epochs, batch size `32`

## Practical Constraints And Expectations

- Prompting is narrow. The training data uses ten hardcoded captions, so the model works best on prompts close to those labels.
- Sampling is slow on CPU. The decoder runs a `1000` step reverse diffusion loop by default.
- Outputs stay in the model's normalized image space during sampling helpers. If you want to save or post-process images outside the notebook, you may want to unnormalize them using the dataset mean and standard deviation.
- Git LFS is required if you want to use the shipped checkpoints under [`trained_models/`](./trained_models).

## External Resources

The current README links point to two useful companion resources:

- Medium article: <https://medium.com/correll-lab/dall-e-2-from-scratch-c055bf881b9a>
- Colab notebook: <https://colab.research.google.com/drive/1g4kWCBPaNEq1YBeBB7iwKeU3_3LRcEFf?usp=sharing>

The article provides the conceptual walkthrough and motivation for the architecture. The repository contains the full training scripts and source-level implementation details that the article intentionally leaves out. The notebook is the quickest way to reproduce the sampling flow interactively.
