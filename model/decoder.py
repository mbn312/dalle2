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