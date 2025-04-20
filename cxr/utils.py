import re

import matplotlib.pyplot as plt
import sklearn
import torchmetrics
import timm
from .networks import (
    VinillaCNN,
    DenseNet,
    VGG,
    ResNet,
    EfficientNet,
    EfficientNetV2,
    LinearClassifier,
    NonLinearClassifier,
    VisionTransformer,
    SwinTransformer,
    MedVisiontransformer,
    TimmModel,
    HybridModel,
    )


def get_network(network_name: str, num_classes: int, task="binary"):
    # pattern = re.compile(rf"{network_name}$", flags=re.I)
    # match = [valid_network_name for valid_network_name in networks.__all__ if pattern.match(valid_network_name)]
    # if not match:
    #     raise ValueError(f"Network {network_name} is not supported by us")
    if task == "binary":
        num_classes = 1
    if re.match("^cnn", network_name, flags=re.I):
        return VinillaCNN(num_classes=num_classes)
    elif re.match("^densenet", network_name, flags=re.I):
        return DenseNet(
            num_classes=num_classes,
            hidden_layer_sizes = [512, 128],
            suffix=re.search("(?<=densenet).+", network_name, flags=re.I)[0],
        )
    elif re.match("^resnet", network_name, flags=re.I):
        return ResNet(
            num_classes=num_classes,
            hidden_layer_sizes = [512, 128],
            suffix=re.search("(?<=resnet).+", network_name, flags=re.I)[0],
        )
    elif re.match("^efficientnet_v2", network_name, flags=re.I):
        return EfficientNetV2(
            num_classes=num_classes,
            hidden_layer_sizes = [512, 128],
            suffix=re.search("(?<=efficientnet_v2_).+", network_name, flags=re.I)[0],
        )
    elif re.match("^efficientnet", network_name, flags=re.I):
        return EfficientNet(
            num_classes=num_classes,
            hidden_layer_sizes = [512, 128],
            suffix=re.search("(?<=efficientnet_).+", network_name, flags=re.I)[0],
        )
    elif re.match("^vgg", network_name, flags=re.I):
        return VGG(
            num_classes=num_classes,
            hidden_layer_sizes = [512, 128],
            suffix=re.search("(?<=vgg).+", network_name, flags=re.I)[0],
        )
    elif re.match("^linear", network_name, flags=re.I):
        return LinearClassifier(
            embedding_size=768,
            num_classes=num_classes,
        )
    elif re.match("^nonlinear", network_name, flags=re.I):
        return NonLinearClassifier(
            hidden_layer_sizes = [512, 128],
            num_classes=num_classes,
        )
    elif re.match("^medvit", network_name, flags=re.I):
        return MedVisiontransformer(
            num_classes=num_classes,
            suffix=re.search("(?<=medvit_).+", network_name, flags=re.I)[0],
        )
    elif re.match("^hybrid", network_name, flags=re.I):
        return HybridModel(
            num_classes=num_classes,
            hidden_layer_sizes = [512, 128],
        )
    # elif re.match("^vit", network_name, flags=re.I):
    #     return VisionTransformer(
    #         num_classes=num_classes,
    #         hidden_layer_sizes = [512, 128],
    #         suffix=re.search("(?<=vit_).+", network_name, flags=re.I)[0],
    #     )
    # elif re.match("^swin", network_name, flags=re.I):
    #     return SwinTransformer(
    #         num_classes=num_classes,
    #         hidden_layer_sizes = [512, 128],
    #         suffix=re.search("(?<=swin_).+", network_name, flags=re.I)[0],
    #     )
    elif network_name in timm.list_models(pretrained=True):
        return TimmModel(num_classes=num_classes, model_name=network_name)
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

def plot_curve(x, y, auc, x_label=None, y_label=None, label=None):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, y, label=f'{label} (AUC: %.3f)' % auc, color='black')
    plt.legend(loc='lower right', fontsize=18)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    if x_label:
        plt.xlabel(x_label, fontsize=24)
    if y_label:
        plt.ylabel(y_label, fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True)

# %matplotlib inline
# labels = eval_df[f'{DIAGNOSIS}_value'].values
# predictions = eval_df[f'{DIAGNOSIS}_prediction'].values
# false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(
#     labels,
#     predictions,
#     drop_intermediate=False)
# auc = sklearn.metrics.roc_auc_score(labels, predictions)
# plot_curve(false_positive_rate, true_positive_rate, auc, x_label='False Positive Rate', y_label='True Positive Rate', label=DIAGNOSIS)