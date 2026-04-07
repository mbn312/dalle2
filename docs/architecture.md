# Architecture Deep Dive

This document explains how the repository is wired together at the source level. It is based on the code in this repo plus the two external resources already linked from the root README:

- Medium article: <https://medium.com/correll-lab/dall-e-2-from-scratch-c055bf881b9a>
- Colab notebook: <https://colab.research.google.com/drive/1g4kWCBPaNEq1YBeBB7iwKeU3_3LRcEFf?usp=sharing>

The article establishes the intended framing: a small DALL-E 2 style `unCLIP` pipeline for Fashion-MNIST. The repository is the implementation of that idea.

## What This Repo Actually Builds

The project trains three separate models:

1. `CLIP`: align images and captions in a shared latent space
2. `DiffusionPrior`: generate a CLIP image embedding from caption information
3. `Decoder`: generate an image from noise while conditioning on the prior output

The decoder is the final text-to-image stage, but it depends on the prior, and the prior depends on CLIP. That is why the training order is fixed and why the decoder constructor loads the prior, which then loads CLIP.

## End-To-End Data Flow

### 1. Dataset And Captioning

[`data/dataset.py`](../data/dataset.py) wraps `torchvision.datasets.FashionMNIST` and assigns one fixed caption per class:

- `An image of a t-shirt/top`
- `An image of trousers`
- `An image of a pullover`
- `An image of a dress`
- `An image of a coat`
- `An image of a sandal`
- `An image of a shirt`
- `An image of a sneaker`
- `An image of a bag`
- `An image of an ankle boot`

Images are resized to `32x32`. Training utilities compute dataset mean and standard deviation dynamically, then normalize the tensors before model training.

### 2. Tokenization

[`data/data_utils.py`](../data/data_utils.py) uses a minimal tokenizer:

- prepend start-of-text with `chr(2)`
- append end-of-text with `chr(3)`
- zero-pad to `text_seq_length`
- encode the string as raw UTF-8 bytes

This means the text stack is byte-level, not BPE- or wordpiece-based. It is simple and works for the tiny closed caption set used here, but it is not a general-purpose text front end.

### 3. CLIP Stage

[`model/clip.py`](../model/clip.py) contains three modules:

- `TextEncoder`
- `ImageEncoder`
- `CLIP`

The image path is a small Vision Transformer:

- convolutional patch embedding
- learned class token
- learned positional embeddings
- stacked Transformer blocks
- final projection into the shared latent space

The text path is a standard Transformer encoder:

- token embedding lookup
- learned positional embeddings
- stacked Transformer blocks
- final layer norm
- take the end-of-text token representation
- project to the shared latent space

Both embeddings are L2-normalized and trained with a symmetric contrastive objective over the batch. This is the embedding space consumed by the later stages.

### 4. Diffusion Prior

[`model/prior.py`](../model/prior.py) defines `DiffusionPrior`. It begins by loading and freezing the trained CLIP model.

The prior builds a very short causal Transformer sequence of length five:

1. caption bytes padded or truncated to `latent_dim`
2. CLIP text embedding
3. timestep embedding
4. noisy CLIP image embedding
5. learned query embedding

That sequence is passed through a decoder-only Transformer with a causal mask. The final token is projected back into the CLIP latent space and trained with MSE against the clean CLIP image embedding.

At sampling time the prior:

- encodes the caption with frozen CLIP
- generates two candidate image embeddings from pure noise
- picks the one with higher dot product against the text embedding

That "best of two" heuristic is a small, repo-specific simplification rather than a canonical DALL-E 2 sampling procedure.

### 5. Diffusion Decoder

[`model/decoder.py`](../model/decoder.py) defines a UNet-style denoiser conditioned on the prior output.

Conditioning enters the decoder in two forms:

- residual blocks receive a combined embedding from diffusion timestep plus projected CLIP image embedding
- attention blocks receive concatenated text features plus learned image tokens derived from the sampled CLIP image embedding

The decoder architecture includes:

- input convolution
- encoder path with residual blocks, downsampling, and selective attention
- bottleneck residual and attention blocks
- decoder path with skip connections and upsampling
- output convolution predicting diffusion noise

The training objective is standard epsilon prediction:

- sample a timestep
- add forward-process noise to a real image
- predict the noise
- minimize MSE between predicted and sampled noise

### 6. Reverse Diffusion Sampling

`sample_image` and `sample_plot_image` in [`model/decoder.py`](../model/decoder.py) implement the reverse diffusion loop.

The process is:

1. start from pure Gaussian noise
2. iterate from `t = T - 1` down to `0`
3. predict noise with the decoder
4. compute the next denoised sample using the DDPM update rule
5. clamp the intermediate image to `[-1, 1]`

`sample_plot_image` additionally displays ten evenly spaced snapshots through the reverse process. That behavior matches the intent described in the linked article and the local notebook.

## Core Building Blocks

[`model/transformer.py`](../model/transformer.py) provides reusable pieces used across the repo:

- `SinusoidalPositionalEmbedding`
- `PatchEmbedding`
- `MultiHeadAttention`
- `TransformerBlock`

These utilities are shared by CLIP, the prior, and the decoder text encoder.

## Training Scripts

### `train_clip.py`

Responsibilities:

- build CLIP
- load the Fashion-MNIST training set
- optionally validate on the test set
- optimize with Adam or AdamW
- use warmup plus cosine annealing
- save the best validation checkpoint when validation is enabled

### `train_prior.py`

Responsibilities:

- ensure a CLIP checkpoint exists
- train the prior on top of frozen CLIP embeddings
- optionally validate
- save the best prior checkpoint

### `train_decoder.py`

Responsibilities:

- ensure CLIP and prior checkpoints exist
- train the UNet denoiser
- optionally validate
- optionally sample after each epoch

The notebook in [`notebooks/diffusion_samples.ipynb`](../notebooks/diffusion_samples.ipynb) mirrors this dependency logic by training missing checkpoints before sampling.

## Configuration Surface

[`data/FMNISTConfig.py`](../data/FMNISTConfig.py) is the single source of truth for defaults.

Important top-level settings:

- `latent_dim = 256`
- `img_size = (32, 32)`
- `img_channels = 1`
- `text_seq_length = 64`
- `dataset = "fashion_mnist"`

Nested config classes separate stage-specific settings:

- `CLIPConfig`
- `PriorConfig`
- `DecoderConfig`

That file is where to change:

- checkpoint paths
- batch sizes
- epoch counts
- augmentation flags
- diffusion schedule length
- UNet width and channel ratios

## How The README Links Map To The Code

### Medium Article

The linked article is the high-level narrative version of the project. It explains:

- why the repo uses a DALL-E 2 style `unCLIP` pipeline
- why the project is scoped to Fashion-MNIST
- why the decoder uses a diffusion UNet
- why the author chose specific building blocks such as GroupNorm and SiLU

The article explicitly leaves out some implementation details and full training scripts. Those details live here in the repository.

### Colab Notebook

The linked Colab is represented locally by [`notebooks/diffusion_samples.ipynb`](../notebooks/diffusion_samples.ipynb). In practice, the notebook does two things:

- trains missing checkpoints
- samples each Fashion-MNIST caption and plots the reverse diffusion trajectory

That makes it the easiest interactive entry point for someone trying to reproduce the project behavior quickly.

## Important Limitations

This codebase is best understood as an educational implementation, not a production text-to-image system.

Key limitations:

- Fashion-MNIST is the only implemented dataset
- prompts are tied closely to ten fixed class captions
- the tokenizer is extremely simple
- the decoder operates at one low resolution only
- there is no classifier-free guidance
- there are no cascaded super-resolution stages
- there is no experiment management layer or command-line interface

There are also practical repo-level constraints:

- checkpoints are large and stored under `trained_models/`
- sampling depends on all three checkpoints being available
- default sampling uses `1000` diffusion steps, which is expensive on CPU

## Suggested Reading Order

If you want to understand the code efficiently, read it in this order:

1. [`data/FMNISTConfig.py`](../data/FMNISTConfig.py)
2. [`data/dataset.py`](../data/dataset.py)
3. [`model/transformer.py`](../model/transformer.py)
4. [`model/clip.py`](../model/clip.py)
5. [`model/prior.py`](../model/prior.py)
6. [`model/decoder.py`](../model/decoder.py)
7. [`train_clip.py`](../train_clip.py)
8. [`train_prior.py`](../train_prior.py)
9. [`train_decoder.py`](../train_decoder.py)

That order follows the actual dependency chain in the code.
