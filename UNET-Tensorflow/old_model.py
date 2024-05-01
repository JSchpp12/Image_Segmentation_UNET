import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms as transforms 
from dataset import PASCAL2007Dataset
import numpy as np
from ohe import Ohe

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape

if __name__ == "__main__":
    image_dir = 'data\\train_JPEGImages\\'
    mask_dir = 'data\\train_SegClasses\\'

    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  transforms.ToTensor() ])

    x = PASCAL2007Dataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    img_mask = x['000032']
    img = img_mask['image']
    mask = img_mask['mask']
    mask_shape = mask.shape
   
    y = Ohe(mask, 22)
    mask_ohe = y.one_hot_encoded_mask()
    s = mask_ohe.shape
    s1 = img.shape
    #model = UNET(in_channels=1, out_channels=22)
    model = UNET(out_channels=22)
    img_add = img[None, :, :, :]
    img_add_shape = img_add.shape
    res = model(img_add)
    res_shape = res.shape
    print(res_shape)
    print('--')