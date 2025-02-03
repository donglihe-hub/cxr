import re

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import get_network, get_metrics


class CXRModule(L.LightningModule):
    def __init__(
        self,
        network_name: str,
        metric_names: list[str],
        num_classes: int,
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
        pos_weight=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.pos_weight = pos_weight

        self.model = get_network(network_name, num_classes)
        self.metric_names = metric_names
        self.num_classes = num_classes
        self.train_metrics = get_metrics(metric_names, task="binary", prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        preds = self(x)
        loss = F.binary_cross_entropy_with_logits(preds, y)
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics(torch.sigmoid(preds), y))
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        preds = self(x)
        loss = F.binary_cross_entropy_with_logits(preds, y)
        self.log("val_loss", loss)
        self.val_metrics.update(torch.sigmoid(preds), y)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_start(self):
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        preds = self(x)
        loss = F.binary_cross_entropy_with_logits(preds, y)
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

        optimizer = getattr(optim, self.optimizer_name)(
            self.parameters(), lr=self.lr, pos_weight=self.pos_weight
        )
        scheduler = ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def forward(self, x):
        return self.model(x)
