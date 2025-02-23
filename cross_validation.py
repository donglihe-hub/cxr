import logging

from cxr.data import CXRBinaryDataModule
import lightning as L

from cxr.data import CXRBinaryDataModule
from cxr.model import CXRModule


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


if __name__ == "__main__":
    main()
