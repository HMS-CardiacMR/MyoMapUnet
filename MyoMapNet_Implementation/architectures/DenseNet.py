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
            nn.Linear(in_features=200*net_scale, out_features=200*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=200*net_scale, out_features=200*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=200*net_scale, out_features=100*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=100*net_scale, out_features=50*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=50*net_scale, out_features=out_channels),
        )


    def forward(self, x):
        x = self.T1fitNet(x)

        return x