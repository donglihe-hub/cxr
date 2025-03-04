import datetime
import logging
import math
import os
from pathlib import Path

import lightning as L
import torch
from jsonargparse import auto_cli
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from cxr.data import CXRBinaryDataModule
from cxr.model import CXRModule
from cxr.trainer import Trainer


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "128"

sys_logger = logging.getLogger(__name__)

seed_everything(42)


class Settings:
    def __init__(
        self,
        file_extension: str,
        data_dir: str,
        out_dir: str,
        split_file: str | None,
        train_len: float | None,
        val_len: float | None,
        test_len: float | None,
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
    ):
        self.file_extension = file_extension
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)

        self.split_file = split_file
        if split_file:
            self.train_len = None
            self.val_len = None
            self.test_len = None
        elif not (train_len and val_len and test_len):
            self.train_len = 0.7
            self.val_len = 0.1
            self.test_len = 0.2
        else:
            self.train_len = train_len
            self.val_len = val_len
            self.test_len = test_len

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


def main():
    # parsing yaml file
    settings = auto_cli(Settings)
    
    # data
    data_module = CXRBinaryDataModule(
        file_extension=settings.file_extension,
        data_dir=settings.data_dir,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        train_len=settings.train_len,
        val_len=settings.val_len,
        test_len=settings.test_len,
        split_file=settings.split_file,
        use_pos_weight=settings.use_pos_weight,
    )
    data_module.prepare_data()
    data_module.setup()

    # model
    model = CXRModule(
        network_name=settings.network,
        num_classes=data_module.num_classes,
        metric_names=settings.metrics,
        optimizer_name=settings.optimizer,
        lr=settings.lr,
        pos_weight=data_module.pos_weight,
    )
    # model = torch.compile(model)

    # callbacks
    run_id = f"{settings.network}_{datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')}"
    out_dir = settings.out_dir / run_id
    monitor_metric = f"val_{settings.monitor_metric}"
    mode = "min" if settings.monitor_metric == "loss" else "max"
    callbacks = [
        EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=settings.patience,
        ),
        ModelCheckpoint(
            dirpath=settings.out_dir / "checkpoints",
            filename=f"{{epoch}}-{{{monitor_metric}:.2f}}",
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
        logger = WandbLogger(name="wandb_log", save_dir=out_dir)
    elif settings.logger == "tensorboard":
        logger = TensorBoardLogger(name="tensorboard_log", save_dir=out_dir)
    else:
        logger = None
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=settings.max_epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=min(50, math.ceil(len(data_module.train_dataset.indices) / settings.batch_size)),
    )

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    model = CXRModule.load_from_checkpoint(callbacks[1].best_model_path)
    trainer.test(model, dataloaders=data_module.test_dataloader())

    print(run_id)
    print(data_module.pos_weight)


if __name__ == "__main__":
    main()
