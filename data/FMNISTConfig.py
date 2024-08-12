import torch
from dataclasses import dataclass, field

@dataclass
class CLIPConfig:
    # Vision Transformer
    patch_size:tuple[int,int] = (4,4)
    vit_width:int = 256
    vit_layers:int = 6
    vit_heads:int = 8
    # Text Transformer
    text_width:int = 256
    text_layers:int = 6
    text_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = True
    validate:bool = True
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    get_val_accuracy:bool = False
    model_location:str = "./clip_fmnist.pt"

@dataclass
class PriorConfig:
    # Diffusion
    max_time:int = 1000
    schedule:str = "cosine"
    schedule_offset:float = 0.008
    # Transformer Decoder
    width:int = 256
    n_layers:int = 6
    n_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = False
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 150
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    model_location:str = "./prior_fmnist.pt"

@dataclass
class DecoderConfig:
    # Diffusion
    max_time:int = 1000
    schedule:str = "cosine"
    # UNet
    n_groups:int = 8
    kernel_size:tuple[int, int] = (3,3)
    model_channels:int = 32
    cond_channels:int = 128
    channel_ratios:list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layer_blocks:int = 2
    dropout:float = 0.1
    use_scale_shift:bool = True
    n_heads:int = 8
    stride:int = 2
    down_pool:bool = False
    r_mlp:int = 4
    bias:bool = False
    text_layers:int = 4
    n_img_tokens:int = 4
    # Training
    augment_data:bool = False
    num_workers:int = 0
    batch_size:int = 32
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 100
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    sample_after_epoch:bool = True
    model_location:str = "./decoder_fmnist.pt"

@dataclass
class FMNISTConfig:
    latent_dim:int = 256
    # Dataset Info
    dataset:str = "fashion_mnist"
    data_location:str = "./../datasets"
    img_size:tuple[int,int] = (32,32)
    img_channels:int = 1
    vocab_size:int = 256
    text_seq_length:int = 64
    # Data Augmentation / Normalization
    prob_hflip:float = 0.5
    crop_padding:int = 4
    train_mean:list[float] = field(default_factory=lambda: [0.2855552])
    train_std:list[float] = field(default_factory=lambda: [0.33848408])
    # Training
    train_val_split:tuple[int,int] = (50000, 10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model Configs
    clip = CLIPConfig()
    prior = PriorConfig()
    decoder = DecoderConfig()
