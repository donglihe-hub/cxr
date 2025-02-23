import datetime
import logging
import os

import lightning as L
import torch
from jsonargparse import auto_cli
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from cxr.data import CXRBinaryDataModule
from cxr.model import CXRModule


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger(__name__)

L.seed_everything(42)


class Settings:
    def __init__(
        self,
        file_extension: str,
        data_dir: str,
        out_dir: str,
        split_file: str | None,
        train_len: int | None,
        val_len: int | None,
        test_len: int | None,
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
        self.data_dir = data_dir
        self.out_dir = out_dir

        self.split_file = split_file
        if not split_file and not (train_len and val_len and test_len):
            self.train_len = 0.7
            self.val_len = 0.1
            self.test_len = 0.2
        else:
            self.train_len = None
            self.val_len = None
            self.test_len = None

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
    settings = auto_cli(Settings)

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

    model = CXRModule(
        network_name=settings.network,
        metric_names=settings.metrics,
        optimizer_name=settings.optimizer,
        lr=settings.lr,
        num_classes=data_module.num_classes,
        pos_weight=data_module.pos_weight,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = f"{settings.out_dir}/{run_id}"
    monitor_metric = f"val_{settings.monitor_metric}"
    mode = "min" if settings.monitor_metric == "loss" else "max"
    callbacks = [
        EarlyStopping(monitor=monitor_metric, mode=mode, patience=settings.patience),
        ModelCheckpoint(
            dirpath=out_dir,
            filename="{epoch}-{val_loss:.2f}-{val_auroc:.2f}",
            save_last=True,
            save_top_k=2,
            monitor=monitor_metric,
            mode=mode,
        ),
    ]

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is unavailable")

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
        # log_every_n_steps=5,
    )

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # test_trainer = L.Trainer(
    #     accelerator="gpu",
    #     devices=1,
    # )
    # best model
    model = CXRModule.load_from_checkpoint(callbacks[1].best_model_path)
    trainer.test(model, dataloaders=data_module.test_dataloader())
    from lightning.fabric.utilities.throughput import measure_flops
    with torch.device("meta"):
        import warnings
        with warnings.catch_warnings(action="ignore"):
            model = CXRModule(
                network_name=settings.network,
                metric_names=settings.metrics,
                optimizer_name=settings.optimizer,
                lr=settings.lr,
                num_classes=data_module.num_classes,
                pos_weight=data_module.pos_weight,
            )
        x = torch.randn(1, 3, 224, 224)
    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)
    print(f"fwd_flops: {fwd_flops / 1e8}")

    model_loss = lambda y: y.sum()
    fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)
    print(f"fwd_bwd_flops: {fwd_and_bwd_flops}")


    print(settings.network)
    print(data_module.pos_weight)
    print(run_id)


if __name__ == "__main__":
    main()
