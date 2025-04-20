import re

import lightning as L
from lightning.fabric import Fabric

fabric = Fabric()
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from .utils import get_network, get_metrics


class CXRModule(L.LightningModule):
    def __init__(
        self,
        network_name: str,
        num_classes: int,
        metric_names: list[str],
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model = get_network(network_name, num_classes)
        self.metric_names = metric_names
        self.num_classes = num_classes
        self.train_metrics = get_metrics(metric_names, task="binary", prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics(torch.sigmoid(preds), y))
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        self.val_metrics.update(torch.sigmoid(preds), y)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def on_test_start(self):
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        preds = self(x)
        loss = self.loss_fn(preds, y)
        if "loss" in self.metric_names:
            self.log("test_loss", loss)

        self.test_metrics.update(torch.sigmoid(preds), y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        pattern = re.compile(rf"{self.optimizer_name}$", flags=re.I)
        match = [
            valid_optimizer_name
            for valid_optimizer_name in optim.__all__
            if pattern.match(valid_optimizer_name)
        ]
        if not match:
            raise ValueError(
                f"Optimizer {self.optimizer_name} is not supported by torch.optim"
            )
        self.optimizer_name = match[0]

        optimizer = getattr(optim, self.optimizer_name)(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def forward(self, x):
        return self.model(x)
