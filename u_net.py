import torch
import torch.nn as nn
from torch.nn.modules.conv import ConvTranspose2d
from torchvision.transforms import CenterCrop
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ This block performs two, 3x3 unpadded convolutions, 
        each followed by a ReLu activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            # we will use batch norm, so we set bias to false
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride),
            nn.ReLU(inplace=True)
            )

    def forward(self, X):
        return self.conv(X)

class ContractingBranch(nn.Module):
    """ This branch performs the contraction step, consisting of
        4 layers of 2, 3x3 upadded convolutions, each followed by a
        2x2 maxpooling operation for downsampling.
    """

    def __init__(self, filters=[3,64,128,256,512,1024]):
        super().__init__()
        self.layers = len(filters)-1
        self.encoded = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.layers):
            self.encoded.append(ConvBlock(filters[l], filters[l+1]))

    def forward(self, X):
        downsampled = []
        for enc in self.encoded:
            X = enc(X)
            downsampled.append(X)
            X = self.maxpool(X)

        return downsampled

class ExpandingBranch(nn.Module):
    def __init__(self, filters=[1024, 512, 256, 128, 64]):
        super().__init__()
        self.layers = len(filters)-1
        self.filters = filters
        self.decoded = nn.ModuleList()
        self.upsampled = nn.ModuleList()
    
        for l in range(self.layers-1):
            self.decoded.append(ConvBlock(self.filters[l], self.filters[l+1]))

        for l in range(self.layers-1):
            self.upsampled.append(ConvTranspose2d(self.filters[l], self.filters[l+1], 2, 2))

    def crop_tensor(self, reference, target):
        """ Crops the downsampled, reference tensor to be able
            to concatenate it with the target, upsampled tensor.

            Arguments:
                - reference [tensor]: the downsampled, reference tensor to crop
                - target [tensor]: tensor to append cropped reference to
            Returns:
                - cropped_reference [tensor]: the cropped reference tensor
        """
        _, _, h, w = target.shape
        cropped_reference   = CenterCrop([h, w])(reference)

        return cropped_reference

    def forward(self, X, downsampled):
        for l in range(self.layers):
            X = self.upsampled[l](X)
            ds = self.crop_tensor(downsampled[l], X)
            X = torch.cat([X, ds], dim=1)
            X = self.decoded[l](X)

        return X
        
class UNet(nn.Module):
    pass

if __name__ == '__main__':
    # test on a random tensor
    enc = ContractingBranch()
    # test tensor for encoding
    x = torch.randn(1, 3, 572, 572)
    squished_rubbish = enc(x)

    # test tensor for decoding
    x = torch.rand(1, 1024, 28, 28)
    dec = ExpandingBranch()
    rubbish_bin = dec(x, squished_rubbish[::-1][1:])
    for trash in rubbish_bin:
        print(trash.shape)