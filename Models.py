import torch
import torch.nn.functional as F
from torch import nn


# Binary image segmentation model In:(N, 3, 4H, 4W) Return:(N, 2, H, W)
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 40, 3, padding=1)
        self.conv1d = nn.Conv2d(40, 40, 2, stride=2)
        self.conv2 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv3 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv4d = nn.Conv2d(40, 40, 2, stride=2)
        self.conv5 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv6 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv7d = nn.Conv2d(40, 40, 2, stride=2)
        self.conv8 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv9d = nn.Conv2d(40, 40, 2, stride=2)
        self.conv10 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv11d = nn.Conv2d(40, 40, 2, stride=2)
        self.conv12 = nn.Conv2d(40, 40, 3, padding=1)
        self.conv13d = nn.Conv2d(40, 40, 2, stride=2)
        self.conv14 = nn.Conv2d(40, 40, 1)
        self.conv15u = nn.ConvTranspose2d(40, 40, 2, stride=2)
        self.conv16 = nn.Conv2d(80, 40, 3, padding=1)
        self.conv17u = nn.ConvTranspose2d(40, 40, 2, stride=2)
        self.conv18 = nn.Conv2d(80, 40, 3, padding=1)
        self.conv19u = nn.ConvTranspose2d(40, 40, 2, stride=2)
        self.conv20 = nn.Conv2d(80, 40, 3, padding=1)
        self.conv21u = nn.ConvTranspose2d(40, 40, 2, stride=2)
        self.conv22 = nn.Conv2d(80, 2, 1)


    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1d(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4d(x))
        x = F.relu(self.conv5(x))
        x1 = F.relu(self.conv6(x))
        x2 = F.relu(self.conv7d(x1))
        x2 = F.relu(self.conv8(x2))
        x3 = F.relu(self.conv9d(x2))
        x3 = F.relu(self.conv10(x3))
        x4 = F.relu(self.conv11d(x3))
        x4 = F.relu(self.conv12(x4))
        y = F.relu(self.conv13d(x4))
        y = F.relu(self.conv14(y))
        y = F.relu(self.conv15u(y))
        y = torch.cat([x4, y], dim=1)
        y = F.relu(self.conv16(y))
        y = F.relu(self.conv17u(y))
        y = torch.cat([x3, y], dim=1)
        y = F.relu(self.conv18(y))
        y = F.relu(self.conv19u(y))
        y = torch.cat([x2, y], dim=1)
        y = F.relu(self.conv20(y))
        y = F.relu(self.conv21u(y))
        y = torch.cat([x1, y], dim=1)
        y = self.conv22(y)
        return y










