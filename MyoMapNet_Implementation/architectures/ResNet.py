from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn


net_scale = 2

class RESNET18(nn.Module):
    def __init__(self, out_channels):
        super(RESNET18, self).__init__()

        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x

class RESNET34(nn.Module):
    def __init__(self, out_channels):
        super(RESNET34, self).__init__()

        self.model = resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x

class RESNET50(nn.Module):
    def __init__(self, out_channels):
        super(RESNET50, self).__init__()

        self.model = resnet50(pretrained=False)
        self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(2048, 25600)
        # self.model.fc1 = nn.Linear(in_features=2048, out_features=200 * net_scale)
        # self.model.activation1 = nn.LeakyReLU()
        # self.model.fc2 = nn.Linear(in_features=200 * net_scale, out_features=200 * net_scale)
        # self.model.activation2 = nn.LeakyReLU()
        # self.model.fc3 = nn.Linear(in_features=200 * net_scale, out_features=100 * net_scale)
        # self.model.activation3 = nn.LeakyReLU()
        # self.model.fc4 = nn.Linear(in_features=100 * net_scale, out_features=100 * net_scale)
        # self.model.activation4 = nn.LeakyReLU()
        # self.model.fc5 = nn.Linear(in_features=100 * net_scale, out_features=50 * net_scale)
        # self.model.activation5 = nn.LeakyReLU()
        #
        # self.model.fc6 = nn.Linear(in_features=50 * net_scale, out_features=out_channels)

    def forward(self, x):
        # x1 = self.model(x)
        # x2 = self.model.fc2(x1)
        # x3 = self.model.fc3(x2)
        # x4 = self.model.fc4(x3)
        # x5 = self.model.fc5(x4)
        # output = self.model.fc6(x5)
        x = self.model(x)
        return x

class RESNET101(nn.Module):
    def __init__(self, out_channels):
        super(RESNET101, self).__init__()

        self.model = resnet101(pretrained=False)
        self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(2048, out_channels)

    def forward(self, x):
        x = self.model(x)

        return x

class RESNET152(nn.Module):
    def __init__(self, out_channels):
        super(RESNET152, self).__init__()

        self.model = resnet152(pretrained=False)
        self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(2048, out_channels)
        # self.model.fc1 = nn.Linear(in_features=in_channels, out_features=200 * net_scale)
        # self.model.activation1 = nn.LeakyReLU()
        # self.model.fc2 = nn.Linear(in_features=200 * net_scale, out_features=200 * net_scale)
        # self.model.activation2 = nn.LeakyReLU()
        # self.model.fc3 = nn.Linear(in_features=200 * net_scale, out_features=100 * net_scale)
        # self.model.activation3 = nn.LeakyReLU()
        # self.model.fc4 = nn.Linear(in_features=100 * net_scale, out_features=100 * net_scale)
        # self.model.activation4 = nn.LeakyReLU()
        # self.model.fc5 = nn.Linear(in_features=100 * net_scale, out_features=50 * net_scale)
        # self.model.activation5 = nn.LeakyReLU()
        #
        # self.model.fc6 = nn.Linear(in_features=50 * net_scale, out_features=out_channels)

    def forward(self, x):
        x = self.model(x)
        return x


