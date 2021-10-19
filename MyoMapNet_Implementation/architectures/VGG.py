from torchvision.models import vgg16, vgg19
import torch.nn as nn

def save_models(model, name, epoch):

    # torch.save(model.state_dict(), "model_save_dir/"+name+".model")
    torch.save(model.state_dict(), "/data2/amine/MyoMapNet/model_save_dir/" + name + ".model")
    print("Model saved at epoch {}".format(epoch))

class VGG16(nn.Module):
    def __init__(self, out_channels):
        super(VGG16, self).__init__()

        self.model = vgg16(pretrained=False)
        self.model.features._modules['0'] = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier._modules['6'] = nn.Linear(4096, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x

class VGG19(nn.Module):
    def __init__(self, out_channels):
        super(VGG19, self).__init__()

        self.model = vgg19(pretrained=False)
        self.model.features._modules['0'] = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier._modules['6'] = nn.Linear(4096, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x

