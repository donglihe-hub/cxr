import datetime
import logging
import os

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from cxr.data import CXRBinaryDataModule
from cxr.model import CXRModule


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger(__name__)

L.seed_everything(42)


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
        use_pos_weight=settings["use_pos_weight"],
    )
    data_module.prepare_data()
    data_module.setup()

    model = CXRModule(
        network_name=settings["network"],
        metric_names=settings["metrics"],
        optimizer_name=settings["optimizer"],
        lr=settings["lr"],
        num_classes=data_module.num_classes,
        pos_weight=data_module.pos_weight,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = f"{settings['out_dir']}/{run_id}"
    monitor_metric = f"val_{settings["monitor_metric"]}"
    mode = "min" if settings["monitor_metric"] == "loss" else "max"
    callbacks = [
        EarlyStopping(monitor=monitor_metric, mode=mode, patience=settings["patience"]),
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

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(save_dir=out_dir),
        max_epochs=settings["max_epochs"],
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, datamodule=data_module)
    test_trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
    )

    # best model
    model = CXRModule.load_from_checkpoint(callbacks[1].best_model_path)
    # it has to be written this way to avoid OOM
    test_trainer.test(model=model, dataloaders=data_module.test_dataloader())


if __name__ == "__main__":
    main()
