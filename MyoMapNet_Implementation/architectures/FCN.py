import torch.nn as nn
import torch

def save_models(model, name, epoch):

    # torch.save(model.state_dict(), "model_save_dir/"+name+".model")
    torch.save(model.state_dict(), "/data2/amine/MyoMapNet/model_save_dir/" + name + ".model")
    print("Model saved at epoch {}".format(epoch))

class FCN(nn.Module):
    def __init__(self, in_channels):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)
        self.linear = nn.Linear(in_channels, in_channels)
        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.pool(self.conv2(x)))

        x = self.relu(self.conv3(x))
        x = self.relu(self.pool(self.conv4(x)))

        x = self.relu(self.conv5(x))
        x = self.relu(self.pool(self.conv6(x)))

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.relu(self.deconv5(x))
        x = self.relu(self.deconv6(x))
        x = self.conv7(x)
        # x = self.linear(x)

        return x