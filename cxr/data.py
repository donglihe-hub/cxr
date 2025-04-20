import itertools
import logging
from pathlib import Path

import cv2
import torchvision.transforms as transforms
import lightning as L
import numpy as np
import pandas as pd
import torch
import tqdm
from pydicom.pixels import pixel_array
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CXRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        assert len(images) == len(labels), "Images and labels must have the same length"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

        assert len(embeddings) == len(labels), "Embeddings and labels must have the same length"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]

        return embedding, label

class CXRBinaryDataModule(L.LightningDataModule):
    def __init__(
        self,
        file_extension: str,
        data_dir: str,
        train_len: float | None,
        val_len: float | None,
        test_len: float | None,
        split_file: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        use_pos_weight=False,
    ):
        super().__init__()
        self.file_extension = file_extension
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_classes = 2

        self.split_file = split_file
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len

        self.use_pos_weight = use_pos_weight
        # the image is expected to have 2 dimensions, i.e., grayscale
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(
                    lambda x: x.unsqueeze(0) if x.dim() == 2 else x
                ),  # Add channel dim to the first axis if missing
                v2.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
                ),  # convert image with 1 channel to 3 channels
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        # preprocessing
        self.train_transform = transforms.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.Lambda(
                lambda x: x.unsqueeze(0) if x.dim() == 2 else x
            ),  # Add channel dim to the first axis if missing
            v2.Lambda(
                lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
            ),  # convert image with 1 channel to 3 channels
            v2.AugMix(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[.5], std=[.5])
        ])
        
        self.test_transform = transforms.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.Lambda(
                lambda x: x.unsqueeze(0) if x.dim() == 2 else x
            ),  # Add channel dim to the first axis if missing
            v2.Lambda(
                lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
            ),  # convert image with 1 channel to 3 channels
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[.5], std=[.5])
        ])

    def prepare_data(self):
        """Load data from DICOM files and save images as numpy arrays."""
        ...

    def setup(self, stage=None):
        """Load image data and split them into train, validation, and test sets"""
        if self.file_extension == "dcm":
            data_paths = sorted(self.data_dir.glob("**/*.dcm"))
            images = np.asarray(
                [
                    pixel_array(i)
                    for i in tqdm.tqdm(data_paths, desc="Loading DICOM files")
                ],
                dtype=object,
            )
            labels = np.array([1 if "Abnormal" in str(i) else 0 for i in data_paths])
            _, counts = np.unique(labels, return_counts=True)
            print(
                f"Normal instances: {counts[0]}; Abnormal instances: {counts[1]}"
            )
        elif self.file_extension in ["jpg", "jpeg", "png"]:
            if self.split_file:
                splits = pd.read_csv(self.split_file)
                # hardcoded
                splits.label = (splits.label == "PNEUMONIA").astype(int)
                train_data = splits[["filename", "label"]][splits.split == "train"]
                val_data = splits[["filename", "label"]][splits.split == "val"]
                test_data = splits[["filename", "label"]][splits.split == "test"]
                train_images = np.asarray(
                    [
                        cv2.imread(i)
                        for i in tqdm.tqdm(
                            train_data.filename,
                            desc=f"Loading {self.file_extension} train files",
                        )
                    ],
                    dtype=object,
                )
                val_images = np.asarray(
                    [
                        cv2.imread(i)
                        for i in tqdm.tqdm(
                            val_data.filename,
                            desc=f"Loading {self.file_extension} validation files",
                        )
                    ],
                    dtype=object,
                )
                test_images = np.asarray(
                    [
                        cv2.imread(i)
                        for i in tqdm.tqdm(
                            test_data.filename,
                            desc=f"Loading {self.file_extension} test files",
                        )
                    ],
                    dtype=object,
                )

                images = np.concatenate((train_images, val_images, test_images))
                labels = np.concatenate(
                    (train_data.label, val_data.label, test_data.label)
                )
                dataset = CXRDataset(images, labels, transform=self.transform)

                lengths = [len(train_data), len(val_data), len(test_data)]
                idx_splits = [
                    torch.arange(start=offset - length, end=offset)
                    for offset, length in zip(itertools.accumulate(lengths), lengths)
                ]

                self.train_dataset = Subset(dataset, idx_splits[0])
                self.val_dataset = Subset(dataset, idx_splits[1])
                self.test_dataset = Subset(dataset, idx_splits[2])
            else:
                data_paths = list(self.data_dir.glob(f"**/*.{self.file_extension}"))
                images = np.asarray(
                    [
                        cv2.imread(i)
                        for i in tqdm.tqdm(
                            data_paths[:5], desc=f"Loading {self.file_extension} files"
                        )
                    ],
                    dtype=object,
                )
                labels = np.array(
                    [1 if "PNEUMONIA" in str(i) else 0 for i in data_paths]
                )
                # _, counts = np.unique(labels, return_counts=True)
                # logger.info(f"Normal instances: {counts[0]}; Abnormal instances: {counts[1]}")
        elif self.file_extension in ["npy", "npz"]:
            data_paths = sorted(self.data_dir.glob(f"**/*_general.{self.file_extension}"))
            embeddings = [np.load(i).astype(np.float32) for i in tqdm.tqdm(data_paths, desc="Loading embedding files")]
            labels = np.array([1 if "Abnormal" in str(i) else 0 for i in data_paths])
            _, counts = np.unique(labels, return_counts=True)
            print(
                f"Normal instances: {counts[0]}; Abnormal instances: {counts[1]}"
            )
        elif self.data_dir in ["chestmnist"]:
            import medmnist
            ...

            
        else:
            raise ValueError(f"File format {self.file_extension} is not supported.")

        if self.use_pos_weight:
            self.pos_weight = torch.tensor([counts[0] / counts[1]])
            logger.info(f"Positive weight is set to {self.pos_weight}")
        else:
            self.pos_weight = None

        if not self.split_file:
            if self.file_extension in ["npy", "npz"]:
                dataset = EmbeddingDataset(embeddings, labels)
                self.test_dataset, self.val_dataset, self.train_dataset = random_split(
                    dataset=dataset,
                    lengths=[self.test_len, self.val_len, self.train_len],
                    generator=torch.Generator().manual_seed(42),
                )
            else:
                dataset = CXRDataset(images, labels, transform=self.test_transform)
                self.test_dataset, self.val_dataset, self.train_dataset = random_split(
                    dataset=dataset,
                    lengths=[self.test_len, self.val_len, self.train_len],
                    generator=torch.Generator().manual_seed(0),
                )
                # 0, 42, 1024, 65536, 4294967296
                self.train_dataset.transform = self.train_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
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
    dm = CXRBinaryDataModule("dcm", data_dir="../data/XR_DICOM")
    dm.prepare_data()
    # dm.save_samples()

    # mean, std = compute_mean_std_gray(dm)
    # print(f"mean: {mean}, std: {std}")


if __name__ == "__main__":
    main()
