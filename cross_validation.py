import datetime
import logging
import math
import os
from pathlib import Path
from collections import defaultdict

import lightning as L
import numpy as np
import torch
from jsonargparse import auto_cli
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from cxr.data import CXRBinaryDataModule, CrossValidationCXRBinaryDataModule
from cxr.model import CXRModule
from cxr.trainer import Trainer

from sklearn.model_selection import KFold

os.environ["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "128"
sys_logger = logging.getLogger(__name__)
seed_everything(42)

class Settings:
    def __init__(
        self,
        file_extension: str,
        data_dir: str,
        out_dir: str,
        num_workers: int = 4,
        batch_size: int = 32,
        network: str = "densenet121",
        optimizer: str = "adam",
        lr: float = 0.001,
        use_pos_weight: bool = False,
        max_epochs: int = 120,
        metrics: list = [],
        monitor_metric: str = "loss",
        patience: int = 60,
        logger="tensorboard",
        n_splits: int = 5,
    ):
        self.file_extension = file_extension
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.network = network
        self.optimizer = optimizer
        self.lr = lr
        self.use_pos_weight = use_pos_weight
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.logger = logger
        self.n_splits = n_splits

def main():
    # parsing yaml or CLI
    settings = auto_cli(Settings)

    # Prepare KFold
    scores = defaultdict(list)

    data_module = CrossValidationCXRBinaryDataModule(
        file_extension=settings.file_extension,
        data_dir=settings.data_dir,
        batch_size=settings.batch_size,
        num_splits=settings.n_splits,
        num_workers=settings.num_workers,
        use_pos_weight=settings.use_pos_weight,
    )
    data_module.prepare_data()
    data_module.setup()
    run_id = f"cv_{settings.network}_{settings.use_pos_weight}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    out_dir = settings.out_dir / run_id
    # callbacks
    monitor_metric = f"val_{settings.monitor_metric}"
    mode = "min" if settings.monitor_metric == "loss" else "max"

    for k in range(settings.n_splits):
        # Create new instance of the DataModule for this fold

        # Model
        model = CXRModule(
            network_name=settings.network,
            num_classes=data_module.num_classes,
            metric_names=settings.metrics,
            optimizer_name=settings.optimizer,
            lr=settings.lr,
            pos_weight=data_module.pos_weight,
        )

        callbacks = [
            ModelCheckpoint(
                dirpath=out_dir / "checkpoints",
                filename=f"{{epoch}}-{{{monitor_metric}:.2f}}_fold{k}",
                save_last=True,
                save_top_k=1,
                monitor=monitor_metric,
                mode=mode,
            ),
        ]

        if not torch.cuda.is_available():
            raise RuntimeError("Cuda is unavailable")

        # logger
        if settings.logger == "wandb":
            logger = WandbLogger(name=run_id, save_dir=out_dir)
        elif settings.logger == "tensorboard":
            logger = TensorBoardLogger(name=run_id, save_dir=out_dir)
        else:
            logger = None

        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            logger=logger,
            max_epochs=settings.max_epochs,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            deterministic=True,
            log_every_n_steps=28,
        )
        torch.use_deterministic_algorithms(True, warn_only=True)

        data_module.update_split(k)
        # Training
        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )
        val_results = trainer.validate(model, dataloaders=data_module.val_dataloader())[0]
        for k, result in val_results.items():
            scores[k].append(result)

    print(run_id)
    print(settings.use_pos_weight)
    for k, scores in scores.items():
        print(f"metric {k}:{np.mean(scores)}, {np.std(scores)}")
    

if __name__ == "__main__":
    main()