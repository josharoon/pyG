""" Simple UNet Segmentation Model """
import torch
import torch.nn as nn


# define a conv block
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
#define an encoder block
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p
#define a decoder block
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0 )
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #encoder
        self.encoder1 = encoder_block(in_channels, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)
        #bottleneck
        self.bottleneck = conv_block(512, 1024)
        #decoder
        self.decoder4 = decoder_block(1024, 512)
        self.decoder3 = decoder_block(512, 256)
        self.decoder2 = decoder_block(256, 128)
        self.decoder1 = decoder_block(128, 64)
        #classification layer
        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        #encoder
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)
        #bottleneck
        x = self.bottleneck(p4)
        #decoder
        x = self.decoder4(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        #classifier
        x = self.conv(x)
        return x