import datetime
import logging
import os

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from cxr.data import CXRBinaryDataModule
from cxr.model import CXRModule


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger(__name__)


def main():
    with open("settings.yml") as f:
        settings = yaml.safe_load(f)
    logger.info(settings)

    data_module = CXRBinaryDataModule(
        data_dir=settings["data_dir"],
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        train_len=settings["train_len"],
        val_len=settings["val_len"],
        test_len=settings["test_len"],
    )

    model = CXRModule(
        network_name=settings["network"],
        metric_names=settings["metrics"],
        optimizer_name=settings["optimizer"],
        lr=settings["lr"],
        num_classes=data_module.num_classes,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = f"{settings['out_dir']}/{run_id}"
    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename="{epoch}-{val_loss:.2f}-{val_auroc:.2f}",
            save_last=True,
            save_top_k=2,
            monitor=f"val_{settings["monitor_metric"]}",
            mode="min" if settings["monitor_metric"] == "loss" else "max",
        ),
    ]
    
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is unavailable")
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(save_dir=out_dir),
        max_epochs=settings["max_epochs"],
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()
