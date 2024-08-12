import torch
import numpy as np

def tokenizer(text, mask=None, text_seq_length=128):
    # If a mask is not inputted it encodes text, otherwise it decodes the text
    if mask is None:
        # Add SOT and EOT tokens
        out = chr(2) + text + chr(3)

        # Pad to inputted sequence length
        out = out + "".join([chr(0) for _ in range(text_seq_length - len(out))])

        # Encode text
        out = torch.IntTensor(list(out.encode("utf-8")))

        # Create text mask
        mask = (out != 0).type(torch.IntTensor)
    else:
        # Decode text
        out = "".join([chr(x) for x in text[1 : (len(mask.nonzero()) - 1)]])
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
def get_beta_schedule(schedule, max_time, s=0.008):
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
        Exception("Beta schedule not implemented.")

    return betas

def get_schedule_values(config):
        schedule_values = {}
        schedule_values["betas"] = get_beta_schedule(config.prior.schedule, config.prior.max_time).to(config.device)
        schedule_values["alphas"] = 1.0 - schedule_values["betas"]
        schedule_values["alpha_bars"] = torch.cumprod(schedule_values["alphas"], axis = 0)
        schedule_values["sqrt_recip_alphas"] = torch.sqrt(1.0 / schedule_values["alphas"])
        schedule_values["sqrt_alpha_bars"] = torch.sqrt(schedule_values["alpha_bars"])
        schedule_values["sqrt_one_minus_alpha_bars"] = torch.sqrt(1.0 - schedule_values["alpha_bars"])
        schedule_values["alpha_bars_prev"] = torch.cat((torch.ones(1, device=config.device), schedule_values["alpha_bars"][:-1]))
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