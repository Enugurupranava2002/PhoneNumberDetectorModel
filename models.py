from regex import F
from torch import amax
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()     ##input will be (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 64, 3) ##output size be (64, 26, 26)
        self.maxPool1 = nn.MaxPool2d(2, 2)##output size be(64, 13, 13)
        self.conv2 = nn.Conv2d(64, 64, 3) ##output size be (64, 11, 11)
        self.mpool1 = nn.MaxPool2d(2,2) ##output size be (64, 5, 5)
        # self.conv3 = nn.Conv2d(64, 40, 3)##output size be (40, 22, 22)
        # self.conv4 = nn.Conv2d(40, 20, 4)##output size be (20, 19, 19)
        # self.conv5 = nn.Conv2d(20, 10, 4)##output size be (10, 16, 16)
        # self.conv6 = nn.Conv2d(10, 2, 4)##output size be (1, 13, 13)
        self.fc1 = nn.Linear(25*64, 80)
        self.fc1_drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(80,10)

    def forward(self, x):
        x = torch.reshape(x, (-1,1,28,28))
        x = self.maxPool1(F.relu(self.conv1(x)))
        # print("After max pool1: ", x.size())
        x = self.mpool1(F.relu(self.conv2(x)))
        # print("After mpool1: ", x.size())
        # print("After conv3: ", x.size())
        # print("After conv4: ", x.size())
        # print("After conv5: ", x.size())
        # print("After conv6: ", x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1_drop(self.fc1(x))
        x = self.fc2(x)
        soft = nn.Softmax(dim=1)
        x = soft(x)
        return x