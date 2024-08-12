import torch
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