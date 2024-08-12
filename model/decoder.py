import torch.nn as nn

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