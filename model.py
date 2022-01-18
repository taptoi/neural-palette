import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        """
        2D ResBlock with instance norm that downsamples to half of
        the input dimensions
        :param int c_in: Number of input channels
        :param int c_out: Number of output channels
        """
        super(DownBlock, self).__init__()
        self.skip = nn.Sequential()

        # 1x1 conv to adapt the number of channels
        if c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=2),
                nn.InstanceNorm2d(num_features=c_out)
            )
        else:
            self.skip = None
        
        self.main_path = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(num_features=c_out),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=c_out)
            )

    def forward(self, x):
        # init skip path
        if(self.skip is not None):
            skip_path = self.skip(x)
        else:
            skip_path = x # identity
        # main path
        x = self.main_path(x)
        # add skip connection
        x += skip_path
        x = F.leaky_relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out):
        """
        2D Upsampling convolution
        :param int c_in: Number of input channels
        :param int c_out: Number of output channels
        """
        super(UpBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(c_out)
            )
                

    def forward(self, input, skip, palette):
        N = input.shape[0]
        x = input
        if(skip is not None):
            x = torch.hstack((skip, x))
        if(palette is not None):
            x = torch.hstack((palette, x))
        x = self.model(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self):
        """
        Downsampling path of the netword that encodes the features of
        the inputs to a compressed form and provides intermediate outputs
        as skip connections to the decoder / upsampling path
        """
        super(FeatureEncoder, self).__init__()
        self.input = None
        self.conv_out = None
        self.res_block_1_out = None
        self.res_block_2_out = None
        self.res_block_3_out = None
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(num_features=64),
            nn.LeakyReLU()
            )
        self.res_block_1 = DownBlock(64, 128)
        self.res_block_2 = DownBlock(128, 256)
        self.res_block_3 = DownBlock(256, 512)

    def forward(self, x):
        self.input = x
        x = self.conv(x)
        self.conv_out = x
        x = self.res_block_1(x)
        self.res_block_1_out = x
        x = self.res_block_2(x)
        self.res_block_2_out = x
        x = self.res_block_3(x)
        self.res_block_3_out = x
        return x

class RecoloringDecoder(nn.Module):
    def __init__(self, encoder:FeatureEncoder, num_colors=6):
        """
        Upsampling path of the network that takes as inputs
        the target palette and the encoded features and produces
        a recolored output matching to the target palette.
        """
        super(RecoloringDecoder, self).__init__()

        self.encoder = encoder
        self.num_color_chs = num_colors * 3
        self.up_block_1 = UpBlock(512 + self.num_color_chs, 256)
        self.up_block_2 = UpBlock(512, 128)
        self.up_block_3 = UpBlock(256 + self.num_color_chs, 64)
        self.up_block_4 = UpBlock(128 + self.num_color_chs, 32)
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=32 + 1, out_channels=2, kernel_size=3, padding=1),
                nn.Tanh()
            )

    def forward(self, input, palette):
        x = input
        h = x.shape[2]
        w = x.shape[3]
        # reshape palette:
        palette_pixel = palette.reshape(-1, palette.shape[1] * palette.shape[2] * palette.shape[3], 1, 1)
        # repeat palette pixel for height and width of the input:        
        palette = palette_pixel.repeat((1, 1, h, w))
        x = self.up_block_1.forward(x, None, palette)
        x = self.up_block_2.forward(x, self.encoder.res_block_2_out, None)
        # update palette dims to match input
        palette = palette_pixel.repeat((1, 1, x.shape[2], x.shape[3]))
        x = self.up_block_3.forward(x, self.encoder.res_block_1_out, palette)
        palette = palette_pixel.repeat((1, 1, x.shape[2], x.shape[3]))
        x = self.up_block_4.forward(x, self.encoder.conv_out, palette)
        
        # append LAB lightness from input before the final convolution
        ll = self.encoder.input[:, 0, :, :].reshape(-1, 1, self.encoder.input.shape[2], self.encoder.input.shape[3])
        x = torch.hstack((ll, x))
        x = self.conv(x)
        return x


class PaletteNet(nn.Module):
    def __init__(self):
        """
        The main model of the PaletteNet with a feature
        encoder and a recoloring decoder
        """
        super(PaletteNet, self).__init__()
        self.encoder = FeatureEncoder()
        self.decoder = RecoloringDecoder(self.encoder)

    def forward(self, x, palette):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x, palette)
        return x

