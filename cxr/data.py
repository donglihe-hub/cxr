import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
import tqdm
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CXRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CXRBinaryDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        train_len: float = 0.7,
        val_len: float = 0.1,
        test_len: float = 0.2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_classes = 2

        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len

        # the image is expected to have 2 dimensions, i.e., grayscale
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(
                    lambda x: x.unsqueeze(0) if x.dim() == 2 else x
                ),  # Add channel dim if missing
                v2.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
                ),  # convert image with 1 channel to 3 channels
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def prepare_data(self):
        """Load data from numpy arrays or DICOM files and save as numpy arrays."""
        if (self.data_dir / "normal_dcm_arrays.npy").exists() and (
            self.data_dir / "abnormal_dcm_arrays.npy"
        ).exists():
            logger.info("Loading data from numpy arrays")
            self.normal_dcm_arrays = np.load(
                self.data_dir / "normal_dcm_arrays.npy", allow_pickle=True
            )
            self.abnormal_dcm_arrays = np.load(
                self.data_dir / "abnormal_dcm_arrays.npy", allow_pickle=True
            )
        else:
            logger.info("Loading data from DICOM files")
            dcm_paths = list(self.data_dir.glob("**/*.dcm"))
            normal_dcm_paths = [i for i in dcm_paths if "Normal" in str(i)]
            abnormal_dcm_paths = [i for i in dcm_paths if "Abnormal" in str(i)]
            self.normal_dcm_arrays = np.asarray(
                [
                    dcmread(i).pixel_array
                    for i in tqdm.tqdm(normal_dcm_paths, desc="Loading Normal images")
                ],
                dtype=object,
            )
            self.abnormal_dcm_arrays = np.asarray(
                [
                    dcmread(i).pixel_array
                    for i in tqdm.tqdm(
                        abnormal_dcm_paths, desc="Loading Abnormal images"
                    )
                ],
                dtype=object,
            )
            np.save(self.data_dir / "normal_dcm_arrays.npy", self.normal_dcm_arrays)
            np.save(self.data_dir / "abnormal_dcm_arrays.npy", self.abnormal_dcm_arrays)

        logger.info(
            f"Normal instances: {len(self.normal_dcm_arrays)}; Abnormal instances: {len(self.abnormal_dcm_arrays)}"
        )

    def setup(self, stage=None):
        """Split data into train, val, and test sets"""
        images = np.concatenate([self.normal_dcm_arrays, self.abnormal_dcm_arrays])
        labels = np.concatenate(
            [
                np.zeros(len(self.normal_dcm_arrays)),
                np.ones(len(self.abnormal_dcm_arrays)),
            ]
        )

        dataset = CXRDataset(images, labels, transform=self.transform)
        self.test_dataset, self.val_dataset, self.train_dataset = random_split(
            dataset=dataset,
            lengths=[self.test_len, self.val_len, self.train_len],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def save_samples(self, sample_size=100):
        from matplotlib import pyplot as plt

        if not getattr(self, "train_dataset", None):
            self.setup()

        def __save_samples(split):
            out_path = Path(self.data_dir / split)
            out_path.mkdir(exist_ok=True)
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(256),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.ToPILImage(),
                ]
            )
            arr = (
                self.normal_dcm_arrays
                if split == "normal"
                else self.abnormal_dcm_arrays
            )
            for idx, i in enumerate(
                tqdm.tqdm(
                    arr[np.random.choice(len(arr), size=sample_size, replace=False)],
                    desc=f"Saving {split} images to disk",
                )
            ):
                plt.imshow(transform(i), cmap="gray")
                plt.axis("off")
                plt.savefig(
                    out_path / f"{split}_{idx}.png", bbox_inches="tight", pad_inches=0
                )

        __save_samples("normal")
        __save_samples("abnormal")


def compute_mean_std_gray(data_module: L.LightningDataModule):
    # calculate mean and standard deviation for normalization for gray-scale images
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for image, _ in data_module.train_dataloader():
        image = image.view(image.size(0), -1)
        num_pixels += image.numel()
        mean += image.mean()
        std += image.std()
    mean /= num_pixels
    std /= num_pixels
    return mean, std


def main():
    dm = CXRBinaryDataModule(data_dir="../data/XR_DICOM")
    dm.prepare_data()
    dm.save_samples()
    # mean, std = compute_mean_std_gray(dm)
    # print(f"mean: {mean}, std: {std}")


if __name__ == "__main__":
    main()
