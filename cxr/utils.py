import re

import torchmetrics
from .networks import SimpleCNNBaseline, DenseNet, VGG, ResNet, EfficientNet


def get_network(network_name: str, num_classes: int, task="binary"):
    # pattern = re.compile(rf"{network_name}$", flags=re.I)
    # match = [valid_network_name for valid_network_name in networks.__all__ if pattern.match(valid_network_name)]
    # if not match:
    #     raise ValueError(f"Network {network_name} is not supported by us")
    if task == "binary":
        num_classes = 1
    if re.match("cnn", network_name, flags=re.I):
        return SimpleCNNBaseline(num_classes=num_classes)
        # return SimpleCNNBaseline(network_config["image_size"])
    elif re.match("densenet", network_name, flags=re.I):
        return DenseNet(
            num_classes=num_classes,
            suffix=re.search("(?<=densenet).+", network_name, flags=re.I)[0],
        )
    elif re.match("resnet", network_name, flags=re.I):
        return ResNet(
            num_classes=num_classes,
            suffix=re.search("(?<=resnet).+", network_name, flags=re.I)[0],
        )
    elif re.match("efficientnet", network_name, flags=re.I):
        return EfficientNet(
            num_classes=num_classes,
            suffix=re.search("(?<=efficientnet_).+", network_name, flags=re.I)[0],
        )
    elif re.match("vgg", network_name, flags=re.I):
        return VGG(
            num_classes=num_classes, suffix=re.search("(?<=vgg).+", network_name, flags=re.I)[0]
        )
    else:
        raise ValueError(f"Network {network_name} is not supported by us")


def get_metrics(metric_names: list[str], task: str, prefix: str):
    """Get classification metrics given the metric names and task type"""
    metric_dict = {}
    for metric_name in metric_names:
        pattern = re.compile(rf"{metric_name}$", flags=re.I)
        match = [
            valid_metric_name
            for valid_metric_name in torchmetrics.classification.__all__
            if pattern.match(valid_metric_name)
        ]
        if not match:
            raise ValueError(f"Metric {metric_name} is not supported by torchmetrics")
        metric_dict[match[0].lower()] = getattr(torchmetrics.classification, match[0])(
            task=task
        )

    metrics = torchmetrics.MetricCollection(metric_dict, prefix=prefix)
    return metrics
