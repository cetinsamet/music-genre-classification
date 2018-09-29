import torch
torch.manual_seed(123)
from torch.nn import Module, Conv2d, MaxPool2d, Linear
import torch.nn.functional as F


class genreNet(Module):

    def __init__(self):
        super(genreNet, self).__init__()

        self.conv1  = Conv2d(in_channels=1,     out_channels=64,    kernel_size=3,  stride=1,   padding=1)
        self.pool1  = MaxPool2d(kernel_size=2)

        self.conv2  = Conv2d(in_channels=64, out_channels=128,      kernel_size=3,  stride=1,   padding=1)
        self.pool2  = MaxPool2d(kernel_size=2)

        self.conv3 = Conv2d(in_channels=128, out_channels=256,      kernel_size=3,  stride=1,   padding=1)
        self.pool3 = MaxPool2d(kernel_size=4)

        self.conv4 = Conv2d(in_channels=256, out_channels=512,      kernel_size=3,  stride=1,   padding=1)
        self.pool4 = MaxPool2d(kernel_size=4)

        self.fc1    = Linear(in_features=2048, out_features=10)

    def forward(self, inp):
        x   = F.elu(self.conv1(inp))
        x   = self.pool1(x)
        #print("After Pool_Layer_1: ", x.size())

        x   = F.elu(self.conv2(x))
        x   = self.pool2(x)
        #print("After Pool_Layer_2: ", x.size())

        x   = F.elu(self.conv3(x))
        x   = self.pool3(x)
        #print("After Pool_Layer_3: ", x.size())

        x   = F.elu(self.conv4(x))
        x   = self.pool4(x)
        #print("After Pool_Layer_4: ", x.size())

        x   = x.view(x.size()[0], -1)
        x   = F.softmax(self.fc1(x))
        return x
