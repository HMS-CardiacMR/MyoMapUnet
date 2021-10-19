import torch.nn as nn

def save_models(model, name, epoch):

    # torch.save(model.state_dict(), "model_save_dir/"+name+".model")
    torch.save(model.state_dict(), "/data2/amine/MyoMapNet/model_save_dir/" + name + ".model")
    print("Model saved at epoch {}".format(epoch))

def conv_block(in_channels, out_channels):

    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.conv5 = conv_block(64, 64)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0))


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        output = self.conv6(conv5)

        return output