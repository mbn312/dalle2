import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data.data_utils import freeze_model, get_schedule_values, extract_and_expand, tokenizer
from model.prior import DiffusionPrior
from model.transformer import SinusoidalPositionalEmbedding, TransformerBlock

class Downsample(nn.Module):
    def __init__(self, n_channels, kernel_size=(3,3), stride=2, down_pool=False):
        super().__init__()

        if down_pool:
            self.down = nn.AvgPool2d(stride)
        else:
            self.down = nn.Conv2d(n_channels, n_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        x = self.down(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, d_in, d_out, kernel_size=(3,3)):
        super().__init__()

        self.conv = nn.Conv2d(d_in, d_out, kernel_size, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, cond_channels=128, n_groups=8, kernel_size=(3,3), dropout=0.0, use_scale_shift=True):
        super().__init__()

        self.use_scale_shift = use_scale_shift

        # Layers up to where input should be conditioned
        self.layers1 = nn.Sequential(
            nn.GroupNorm(n_groups, d_in),
            nn.SiLU(),
            nn.Conv2d(d_in, d_out, kernel_size, padding=1)
        )

        # Activation & Linear Projection for Embedding
        self.cond_layers = nn.Sequential(
            nn.SiLU(),
            # d_out multiplied by 2 in order to split into scale & shift if necessary
            nn.Linear(cond_channels, d_out * 2 if use_scale_shift else d_out)
        )

        # Layers after conditioning
        self.layers2 = nn.Sequential(
            nn.GroupNorm(n_groups, d_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(d_out, d_out, kernel_size, padding=1),
            nn.GroupNorm(n_groups, d_out)
        )

        # If necessary, applies convolution to original input to match channels to output 
        self.residual = nn.Conv2d(d_in, d_out, 1) if d_in != d_out else nn.Identity()

    def forward(self, x_0, emb):
        x = self.layers1(x_0)

        emb = self.cond_layers(emb)

        # Adding dimensions to embedding
        while len(emb.shape) < len(x.shape):
            emb = emb[..., None]

        # Conditioning input with embedding
        if self.use_scale_shift:
            # Getting scale and shift
            y_s, y_b = emb.chunk(2, dim=1)
            # Performing scale and shift
            x = y_s * x + y_b
        else:
            # Adding embedding to input
            x += emb

        x = self.layers2(x)

        # Skip Connection
        x += self.residual(x_0)

        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, n_channels, cond_channels, n_groups=8, n_heads=1, dropout=0.0):
        super().__init__()

        assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_size = n_channels // n_heads
        self.scale = self.head_size ** -0.5

        self.group_norm = nn.GroupNorm(n_groups, n_channels)

        # Linear layer for input
        self.qkv = nn.Linear(n_channels, n_channels * 3)

        # Linear layer for conditioning information
        self.cond_kv = nn.Linear(cond_channels, n_channels * 2)

        # Output projection
        self.out_proj = nn.Linear(n_channels, n_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def attention(self, x, cond=None):
        B, L, _ = x.shape

        # Getting queries, keys, and values for input
        Q, K, V = self.qkv(x).chunk(3, dim=-1) # (B, L, C) -> (B, L, C), (B, L, C), (B, L, C)

        Q = Q.view(B, L, self.n_heads, self.head_size).transpose(1, 2) # (B, L, C) -> (B, n_heads, L, head_size)

        K = K.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

        V = V.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

        if cond is not None:
            # Getting keys and values from conditioning information
            k_c, v_c = self.cond_kv(cond).chunk(2, dim=-1) # (B, L_c, C) -> (B, L_c, C), (B, L_c, C)
            k_c = k_c.view(B, cond.shape[1], self.n_heads, self.head_size).transpose(1, 2) # (B, L_c, C) -> (B, n_heads, L_c, head_size)
            v_c = v_c.view(B, cond.shape[1], self.n_heads, self.head_size).transpose(1, 2)

            # Concatenating keys and values of condition to keys and values of input
            K = torch.cat((K, k_c), dim=-2) # (B, n_heads, L, head_size), (B, n_heads, L_c, head_size) -> (B, n_heads, L_xc, head_size)
            V = torch.cat((V, v_c), dim=-2)

        # Get dot product between queries and keys
        attention = torch.matmul(Q, K.transpose(-2, -1)) # (B, n_heads, L, head_size) @ (B, n_heads, head_size, L_xc) -> (B, n_heads, L, L_xc)

        # Scale
        attention = attention * self.scale

        # Applying softmax
        attention = torch.softmax(attention, dim=-1)

        # Get dot product with values
        attention = torch.matmul(attention, V) # (B, n_heads, L, L_xc) @ (B, n_heads, L_xc, head_size) -> (B, n_heads, L, head_size)

        # Combine heads
        attention = attention.transpose(1, 2) # (B, n_heads, L, head_size) -> (B, L, n_heads, head_size)
        attention = attention.contiguous().view(x.shape) # (B, L, n_heads, head_size) -> (B, L, C)

        # Output projection
        attention = self.out_proj(attention) # (B, L, C) -> (B, L, C)

        # Dropout
        attention = self.dropout(attention)

        return attention

    def forward(self, x_0, cond=None):
        b, c, h, w = x_0.shape

        # Group normalization
        x = self.group_norm(x_0)

        # Changing shape to perform attention
        x = x.permute(0, 2, 3, 1).view(b, h * w, c) # (B, C, H, W) -> (B, L, C)

        # Attention
        x = self.attention(x, cond)

        # Changing back to original shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2) # (B, L, C) -> (B, C, H, W)

        # Residual connection
        x = x + x_0

        return x
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Loading Prior Model
        self.prior = DiffusionPrior(config).to(config.device)
        self.prior.load_state_dict(torch.load(config.prior.model_location, map_location=config.device))
        freeze_model(self.prior)

        # MLP to get time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(config.decoder.max_time, config.decoder.model_channels),
            nn.Linear(config.decoder.model_channels, config.decoder.cond_channels),
            nn.SiLU(),
            nn.Linear(config.decoder.cond_channels, config.decoder.cond_channels)
        )

        # MLP to project CLIP image embeddings
        self.img_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder.cond_channels),
            nn.SiLU(),
            nn.Linear(config.decoder.cond_channels, config.decoder.cond_channels)
        )

        # Projection to get image tokens
        self.get_img_tokens = nn.Linear(1, config.decoder.n_img_tokens)

        # Embedding layer for text captions
        self.text_embedding = nn.Embedding(config.vocab_size, config.latent_dim)

        # Learned positional encodings for text captions
        self.positional_encodings = nn.Parameter(torch.randn(config.text_seq_length,config.latent_dim) * (config.latent_dim ** -0.5))

        # Transformer encoder blocks to encode text captions
        self.text_encoder = nn.ModuleList(
            [TransformerBlock(
                config.latent_dim,
                cond_width=config.latent_dim,
                n_heads=config.decoder.n_heads,
                dropout=config.decoder.dropout,
                r_mlp=config.decoder.r_mlp,
                bias=config.decoder.bias
            ) for _ in range(config.decoder.text_layers)]
        )

        # Final layer norm for encoding text captions
        self.final_ln = nn.LayerNorm(config.latent_dim)

        ch = config.decoder.model_channels

        # Initial convolution
        self.in_conv = nn.Conv2d(config.img_channels, ch, config.decoder.kernel_size, padding=1)

        # UNet Encoder Layers
        self.encoder = nn.ModuleList([])
        for r in config.decoder.channel_ratios:
            for _ in range(config.decoder.n_layer_blocks):
                # Add residual block to encoder layer
                self.encoder.append(ResidualBlock(ch, config.decoder.model_channels * r, config.decoder.cond_channels, config.decoder.n_groups, config.decoder.kernel_size, config.decoder.dropout, config.decoder.use_scale_shift))

                # Update number of channels
                ch = config.decoder.model_channels * r

                # Outer encoder layers has no attention blocks
                if r != config.decoder.channel_ratios[0] and r != config.decoder.channel_ratios[-1]:
                    # Add attention block to encoder layer
                    self.encoder.append(AttentionBlock(ch, config.latent_dim, config.decoder.n_groups, config.decoder.n_heads, config.decoder.dropout))

            # No downsample for last encoder layer
            if r != config.decoder.channel_ratios[-1]:
                # Add downsample to encoder layer
                self.encoder.append(Downsample(ch, config.decoder.kernel_size, config.decoder.stride, config.decoder.down_pool))

        # UNet Bottleneck Layers
        self.bottleneck = nn.ModuleList([])
        for block in range(config.decoder.n_layer_blocks):
            # Add residual block to bottleneck layer
            self.bottleneck.append(ResidualBlock(ch, ch, config.decoder.cond_channels, config.decoder.n_groups, config.decoder.kernel_size, config.decoder.dropout, config.decoder.use_scale_shift))

            # No attention block at end of bottleneck layer
            if block != config.decoder.n_layer_blocks - 1:
                # Add attention block to bottleneck layer
                self.bottleneck.append(AttentionBlock(ch, config.latent_dim, config.decoder.n_groups, config.decoder.n_heads, config.decoder.dropout))

        # UNet Decoder Layers
        self.decoder = nn.ModuleList([])
        for r in range(len(config.decoder.channel_ratios))[::-1]:
            for _ in range(config.decoder.n_layer_blocks):
                # Add residual block to decoder layer
                self.decoder.append(ResidualBlock(ch * 2, ch, config.decoder.cond_channels, config.decoder.n_groups, config.decoder.kernel_size, config.decoder.dropout, config.decoder.use_scale_shift))

                # Outer decoder layers has no attention blocks
                if r != 0 and r!= len(config.decoder.channel_ratios) - 1:
                    # Add attention block to decoder layer
                    self.decoder.append(AttentionBlock(ch, config.latent_dim, config.decoder.n_groups, config.decoder.n_heads, config.decoder.dropout))

            # No upsample for last decoder layer
            if r != 0:
                # Update number of channels
                ch = config.decoder.model_channels * config.decoder.channel_ratios[r-1]

                # Add upsample to decoder layer
                self.decoder.append(Upsample(config.decoder.model_channels * config.decoder.channel_ratios[r], ch, config.decoder.kernel_size))

        # Output projection
        self.output = nn.Sequential(
            nn.GroupNorm(config.decoder.n_groups, config.decoder.model_channels),
            nn.SiLU(),
            nn.Conv2d(config.decoder.model_channels, config.img_channels, config.decoder.kernel_size, padding=1)
        )

        # Skip connections
        self.connections = []

    def encode_text(self, text, mask=None):
        x = self.text_embedding(text)

        x = x + self.positional_encodings

        for block in self.text_encoder:
            x = block(x, mask=mask)

        x = self.final_ln(x)

        return x

    def forward(self, x, time, caption=None, mask=None):
        # Sample prior model to get CLIP image embeddings
        img_embeddings = self.prior.sample(caption, mask).to(x.device)

        # Get conditioning information for residual blocks
        c_emb = self.time_mlp(time) + self.img_projection(img_embeddings)

        # Get conditioning information for attention blocks
        c_attn = self.get_img_tokens(img_embeddings[..., None]).permute(0, 2, 1)
        if caption is not None:
            c_attn = torch.cat([self.encode_text(caption, mask), c_attn], dim=1)

        # Initial convolution
        x = self.in_conv(x)

        # UNet encoder layers
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, c_emb)
                # Getting skip connection
                self.connections.append(x)
            elif isinstance(module, AttentionBlock):
                x = module(x, cond=c_attn)
            else:
                x = module(x)

        # UNet bottleneck layers
        for module in self.bottleneck:
            if isinstance(module, ResidualBlock):
                x = module(x, c_emb)
            elif isinstance(module, AttentionBlock):
                x = module(x, cond=c_attn)
            else:
                x = module(x)

        # UNet decoder layers
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                # Concatenate skip connection to end of input
                x = torch.cat([x, self.connections.pop()], dim=1)
                x = module(x, c_emb)
            elif isinstance(module, AttentionBlock):
                x = module(x, cond=c_attn)
            else:
                x = module(x)

        # Output projection
        x = self.output(x)

        return x
    
@torch.no_grad()
def sample_image(config, prompt, mask, schedule_values=None, decoder=None):
    # Load decoder model
    if decoder is None:
      decoder = Decoder(config).to(config.device)
      decoder.load_state_dict(torch.load(config.decoder.model_location, map_location=config.device))
      decoder.eval()

    B = prompt.shape[0]
    # Get completely noisy image
    img = torch.randn((B, config.img_channels, config.img_size[0], config.img_size[1]), device=config.device)

    # Calculate schedule values
    if schedule_values is None:
        schedule_values = get_schedule_values(config)

    for t in range(0, config.decoder.max_time)[::-1]:
        # Setting the timesteps for all the items in the batch
        timesteps = torch.full((B,), t, device=config.device, dtype=torch.long)

        # Getting schedule values for timestep
        sqrt_recip_alphas_t = extract_and_expand(schedule_values["sqrt_recip_alphas"], timesteps, img.shape)
        betas_t = extract_and_expand(schedule_values["betas"], timesteps, img.shape)
        sqrt_one_minus_alpha_bars_t = extract_and_expand(schedule_values["sqrt_one_minus_alpha_bars"], timesteps, img.shape)
        sigma_t = extract_and_expand(schedule_values["sigma"], timesteps, img.shape)

        # Predicting noise at timestep t with decoder
        pred_noise = decoder(img, timesteps, caption=prompt, mask=mask)

        # Generating random noise
        z = torch.randn_like(img) if t > 0 else 0

        # Calculating image at timestep t-1
        img = sqrt_recip_alphas_t * (img - (betas_t / sqrt_one_minus_alpha_bars_t) * pred_noise) + (sigma_t * z)

        img = torch.clamp(img, -1.0, 1.0)

    return img

@torch.no_grad()
def sample_plot_image(config, prompt, mask, schedule_values=None, decoder=None):
    # Load decoder model
    if decoder is None:
        decoder = Decoder(config).to(config.device)
        decoder.load_state_dict(torch.load(config.decoder.model_location, map_location=config.device))

    decoder.eval()

    B = prompt.shape[0]
    # Get completely noisy image
    img = torch.randn((B, config.img_channels, config.img_size[0], config.img_size[1]), device=config.device)

    # Calculate schedule values
    if schedule_values is None:
        schedule_values = get_schedule_values(config)

    plt.figure(figsize=(25,3))
    plt.axis('off')
    num_images = 10
    plot_imgs = torch.linspace(0, config.decoder.max_time-1, 10, dtype=torch.int)

    for t in range(0, config.decoder.max_time)[::-1]:
        # Setting the timesteps for all the items in the batch
        timesteps = torch.full((B,), t, device=config.device, dtype=torch.long)

        # Getting schedule values for timestep
        sqrt_recip_alphas_t = extract_and_expand(schedule_values["sqrt_recip_alphas"], timesteps, img.shape)
        betas_t = extract_and_expand(schedule_values["betas"], timesteps, img.shape)
        sqrt_one_minus_alpha_bars_t = extract_and_expand(schedule_values["sqrt_one_minus_alpha_bars"], timesteps, img.shape)
        sigma_t = extract_and_expand(schedule_values["sigma"], timesteps, img.shape)

        # Predicting noise at timestep t with decoder
        pred_noise = decoder(img, timesteps, caption=prompt, mask=mask)

        # Generating random noise
        z = torch.randn_like(img) if t > 0 else 0

        # Calculating image at timestep t-1
        img = sqrt_recip_alphas_t * (img - (betas_t / sqrt_one_minus_alpha_bars_t) * pred_noise) + (sigma_t * z)

        # Plotting image
        if t == plot_imgs[-1]:
            plot_imgs = plot_imgs[:-1]
            plt.subplot(1, num_images, num_images - len(plot_imgs))
            plt.imshow(img.detach().cpu()[0].permute(1,2,0), cmap="gray" if config.img_channels == 1 else None)

        img = torch.clamp(img, -1.0, 1.0)

    # Add title to plot
    title, _ = tokenizer(prompt[0], mask[0], text_seq_length=config.text_seq_length)
    plt.suptitle(f'Prompt: {title}')
    plt.show()

    return img