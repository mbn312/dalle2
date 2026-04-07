from pathlib import Path

import torch
import numpy as np

def is_lfs_pointer_file(path):
    path = Path(path)
    if not path.is_file():
        return False

    with path.open("rb") as f:
        header = f.read(200)

    try:
        header_text = header.decode("utf-8")
    except UnicodeDecodeError:
        return False

    return header_text.startswith("version https://git-lfs.github.com/spec/v1")

def has_valid_checkpoint(path):
    path = Path(path)
    return path.is_file() and not is_lfs_pointer_file(path)

def load_model_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if is_lfs_pointer_file(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not a real checkpoint. Run `git lfs pull` to fetch model weights."
        )

    return torch.load(path, map_location=device)

def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def tokenizer(text, mask=None, text_seq_length=128):
    # If a mask is not inputted it encodes text, otherwise it decodes the text
    if mask is None:
        # Add SOT and EOT tokens
        out = chr(2) + text + chr(3)
        if len(out) > text_seq_length:
            out = out[: text_seq_length - 1] + chr(3)

        # Pad to inputted sequence length
        out = out + "".join([chr(0) for _ in range(text_seq_length - len(out))])

        # Encode text
        out = torch.LongTensor(list(out.encode("utf-8")))

        # Create text mask
        mask = (out != 0).type(torch.long)
    else:
        if isinstance(text, torch.Tensor):
            text = text.detach().cpu()

        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu()

        # Decode text
        out = "".join([chr(int(x)) for x in text[1 : (len(mask.nonzero()) - 1)]])
        mask = None

    return out, mask

# Gets mean and standard deviation of a set of images
def get_mean_std(data, img_channels, denom=1):
    # Get only the images from the dataset
    try:
        images = np.array([x["image"] for x in data]) / denom
    except:
        images = np.array([x[0] for x in data]) / denom

    # Combine pixels of each channel into one dimension
    images = images.reshape(img_channels, -1)

    # Calculate the mean and standard deviation
    mean, std = images.mean(axis=1), images.std(axis=1)

    return mean, std

# Returns beta schedule
def get_beta_schedule(schedule="linear", max_time=1000, s=0.008):
    if schedule == "linear":
        scale = 1000 / max_time
        betas = torch.linspace(1e-4  * scale, 0.02  * scale, max_time)
    elif schedule == "cosine":
        t = torch.linspace(0, max_time, max_time + 1)
        a_bars = torch.cos((((t / max_time) + s) / (1 + s)) * (np.pi / 2)) ** 2
        a_bars = a_bars / a_bars[0]
        betas = 1 - (a_bars[1:] / a_bars[:-1])
        betas = torch.clamp(betas, min=0, max=0.999)
    else:
        raise ValueError(f"Beta schedule not implemented: {schedule}")

    return betas

def get_schedule_values(
    schedule="linear",
    max_time=1000,
    schedule_offset=0.008,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    schedule_values = {}
    schedule_values["betas"] = get_beta_schedule(schedule, max_time, s=schedule_offset).to(device)
    schedule_values["alphas"] = 1.0 - schedule_values["betas"]
    schedule_values["alpha_bars"] = torch.cumprod(schedule_values["alphas"], dim=0)
    schedule_values["sqrt_recip_alphas"] = torch.sqrt(1.0 / schedule_values["alphas"])
    schedule_values["sqrt_alpha_bars"] = torch.sqrt(schedule_values["alpha_bars"])
    schedule_values["sqrt_one_minus_alpha_bars"] = torch.sqrt(1.0 - schedule_values["alpha_bars"])
    schedule_values["alpha_bars_prev"] = torch.cat((torch.ones(1, device=device), schedule_values["alpha_bars"][:-1]))
    schedule_values["sigma"] = schedule_values["betas"] * (1.0 - schedule_values["alpha_bars_prev"]) / (1.0 - schedule_values["alpha_bars"])
    return schedule_values

def extract_and_expand(x, idx, shape):
    return x[idx].reshape(idx.shape[0], *((1, ) * (len(shape) - 1)))

def freeze_model(model, set_eval=True):
    if set_eval:
        model.eval()

    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def forward_diffusion(x_0, schedule_values, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_bars = extract_and_expand(schedule_values["sqrt_alpha_bars"], t, x_0.shape)
    sqrt_one_minus_alpha_bars = extract_and_expand(schedule_values["sqrt_one_minus_alpha_bars"], t, x_0.shape)
    x_noisy = (sqrt_alpha_bars * x_0) + (sqrt_one_minus_alpha_bars * noise)
    return x_noisy, noise
