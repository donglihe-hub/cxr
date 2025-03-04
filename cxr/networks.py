import torch.nn.functional as F
import torchvision
from torch import nn


__all__ = [
    "VinillaCNN",
    "DenseNet",
    "VGG",
    "EfficientNet",
    "ResNet",
]


class VinillaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, 128 * 28 * 28)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, suffix: int | str = 121, weights="DEFAULT"):
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


class ResNet(nn.Module):
    def __init__(self, num_classes, suffix: str = "18", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["18", "34", "50", "101", "152"]
        self.network = getattr(torchvision.models, f"resnet{suffix}")(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        x = self.network(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes, suffix: str = "b0", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
        self.network = getattr(torchvision.models, f"efficientnet_{suffix}")(
            weights=weights
        )
        self.network.classifier[-1] = nn.Linear(
            self.network.classifier[-1].in_features, num_classes
        )

    def forward(self, x):
        x = self.network(x)
        return x
