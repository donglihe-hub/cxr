import torch.nn.functional as F
import torchvision
from torch import nn


__all__ = [
    "SimpleCNNBaseline",
    "DenseNet",
    "VGG",
]


class SimpleCNNBaseline(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (image_size // 2) * (image_size // 2), 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, suffix: int = 121, weights="DEFAULT"):
        super().__init__()
        assert suffix in ["121", "161", "169", "201"]
        self.network = getattr(torchvision.models, f"densenet{suffix}")(weights=weights)
        self.network.classifier = nn.Linear(
            self.network.classifier.in_features, num_classes
        )

    def forward(self, x):
        x = self.network(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes, suffix: str = "11", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["11", "11_bn", "13", "13_bn", "16", "16_bn", "19", "19_bn"]
        self.network = getattr(torchvision.models, f"vgg{suffix}")(weights=weights)
        self.network.classifier[-1] = nn.Linear(
            self.network.classifier[-1].in_features, num_classes
        )

    def forward(self, x):
        x = self.network(x)
        return x
