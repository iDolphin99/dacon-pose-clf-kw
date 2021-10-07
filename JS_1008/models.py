import torch
import torch.nn as nn
import torch.nn.functional as F

# 1d CNN
class model1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Conv1d(in_channels=6, out_channels=20, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=100, kernel_size=2, stride=2)
        self.bn2 = torch.nn.BatchNorm1d(100)
        self.layer3 = torch.nn.Conv1d(in_channels=100, out_channels=30, kernel_size=2, stride=2)
        self.bn3 = torch.nn.BatchNorm1d(30)

        self.activation = torch.nn.ReLU()

        self.fc = nn.Linear(30 * 75, 61)
        self.bn4 = torch.nn.BatchNorm1d(61)

    def forward(self, x):

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)

        x = x.view(-1, 30 * 75)
        x = self.fc(x)
        x = self.bn4(x)
        x = self.activation(x)

        return x

# Inception
class Inception(nn.Module):
  def __init__(self, in_channel):
    super().__init__()
    # Block 1
    self.branch1_1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1)
    self.branch1_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    # Block 2
    self.branch3_1 = nn.Conv2d(in_channel, 16, kernel_size=1)  # 1x1 Conv
    self.branch3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.branch3_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    # Block 3
    self.branch_pool = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    # print("Input shape : ", np.shape(x)) # torch.Size([100, 8, 301, 4])

    branch1x1 = self.branch1_1(x)
    branch1x1 = self.branch1_2(branch1x1)

    branch3x3 = self.branch3_1(x)
    branch3x3 = self.branch3_2(branch3x3)
    branch3x3 = self.branch3_3(branch3x3)

    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    # 3개의 output들을 1개의 list로
    outputs = [branch1x1, branch3x3, branch_pool]  # np.shape(outputs)) (3,)

    # torch.cat (concatenate)
    cat = torch.cat(outputs, 1)  # outputs list의 tensor들을 dim = 1로 이어준다.

    #cat.shape : torch.Size([300, 32, 301, 4])
    return cat

class model2(nn.Module):
  def __init__(self):
    super().__init__()

    self.Conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
    self.Conv2 = nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1)

    self.Incept1 = Inception(in_channel=8)
    self.Incept2 = Inception(in_channel=16)

    self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.mp2 = nn.MaxPool2d(kernel_size=3, stride=1)

    self.fc1 = nn.Linear(96 * 298 * 1, 3000)
    self.fc2 = nn.Linear(3000, 1000)
    self.fc3 = nn.Linear(1000, 61)

  def forward(self, x):

    out = self.Conv1(x)  # out_channel = 8
    out = F.relu(self.mp1(out))
    out = self.Incept1(out)  # out_channel = 96

    out = self.Conv2(out)  # out_channel = 16
    out = F.relu(self.mp2(out))
    out = self.Incept2(out)  # out_channel = 96

    out = out.view(-1, 96 * 298 * 1)

    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))

    return out

# VGG
class model3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(  # 1 x 600 x 6
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(  # 32 x 300 x 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.fc1 = nn.Linear(64 * 298, 100)  # 512 자리에 512 x width x heigth 넣어주기
        self.fc2 = nn.Linear(100, 61)

    def forward(self, x):
        # Conv layer
        out = self.layer1(x)  # out shape torch.Size([100, 32, 300, 3])
        out = self.layer2(out)  # out shape torch.Size([100, 64, 298, 1])

        # flatten
        out = out.view(-1, 64 * 298)

        # fc layer
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out
