import torch.nn as nn
import torch

net_scale = 2

def save_models(model, name, epoch):

    # torch.save(model.state_dict(), "model_save_dir/"+name+".model")
    torch.save(model.state_dict(), "/data2/amine/MyoMapNet/model_save_dir/" + name + ".model")
    print("Model saved at epoch {}".format(epoch))

class DenseNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseNN, self).__init__()

        self.T1fitNet = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=200*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=200*net_scale, out_features=200*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=200*net_scale, out_features=100*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=100*net_scale, out_features=100*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=100*net_scale, out_features=50*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=50*net_scale, out_features=out_channels),
        )

    def forward(self, x):
        x = self.T1fitNet(x)
        return x

        
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):

    def __init__(self, in_channels):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 =  double_conv(256 + 512, 256)
        self.dconv_up2 =  double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64,  64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out




        