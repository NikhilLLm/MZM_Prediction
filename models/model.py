import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ConductanceResNet(nn.Module):
    def __init__(self):
        super(ConductanceResNet, self).__init__()
        # Use the modern weights API instead of deprecated pretrained
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 2)
        )

    def get_conv_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        conv_features = self.get_conv_features(x)
        x = self.resnet.avgpool(conv_features)
        x = self.flatten(x)
        x = self.resnet.fc(x)
        return x, self.flatten(conv_features)