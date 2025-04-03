from typing import Tuple, List
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch import nn
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from modules.models.autoencoder import AutoencoderKL


def parse_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    elif dtype == "int16":
        return torch.int16
    elif dtype == "int32":
        return torch.int32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError("Invalid dtype")


class IdentityDataset(torch.utils.data.Dataset):
    """
    Simple dataset that returns the same data (d0, d1, ..., dn)
    """

    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]


def normalize(input_data, norm="centered-norm"):
    assert norm in [
        "centered-norm",
        "z-score",
        "min-max",
    ], "Invalid normalization method"

    if norm == "centered-norm":
        norm = lambda x: (2 * x - x.min() - x.max()) / (x.max() - x.min())
    elif norm == "z-score":
        norm = lambda x: (x - x.mean()) / x.std()
    elif norm == "min-max":
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
    return norm(input_data)


class BRATSDataset(torch.utils.data.Dataset):
    """
    Images always as first argument, labels and other variables as last
    Transforms are applied on first argument only
    """

    def __init__(
        self,
        *data,
        transform=None,
        resize=None,
        horizontal_flip=None,
        vertical_flip=None,
        random_crop_size=None,
        rotation=None,
        dtype=torch.float32
    ):
        super().__init__()
        self.data = data

        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(resize) if resize is not None else nn.Identity(),
                    (
                        T.RandomHorizontalFlip(p=horizontal_flip)
                        if horizontal_flip
                        else nn.Identity()
                    ),
                    (
                        T.RandomVerticalFlip(p=vertical_flip)
                        if vertical_flip
                        else nn.Identity()
                    ),
                    (
                        T.CenterCrop(random_crop_size)
                        if random_crop_size is not None
                        else nn.Identity()
                    ),
                    (
                        T.RandomRotation(rotation, fill=-1)
                        if rotation is not None
                        else nn.Identity()
                    ),
                    T.ConvertImageDtype(dtype),
                    # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return self.data[0].__len__()

    def __getitem__(self, index):
        return [
            self.transform(d[index]) if i == 0 else d[index]
            for i, d in enumerate(self.data)
        ]

    def sample(self, n, transform=None):
        """sampling randomly n samples from the dataset, apply or not the transform"""
        transform = self.transform if transform is not None else lambda x: x
        idx = np.random.choice(len(self), n)
        return [
            transform(d[idx]) if i == 0 else d[idx] for i, d in enumerate(self.data)
        ]


class BRATSDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,  # should target the a npy file generated via create_*.py scripts
        train_ratio: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        horizontal_flip=None,
        vertical_flip=None,
        rotation=None,
        random_crop_size=None,
        dtype=torch.float32,
        verbose=True,
        include_radiomics=True,
        **kwargs
    ):
        super().__init__()
        self.dataset_kwargs = {
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "random_crop_size": random_crop_size,
            "rotation": rotation,
            "dtype": dtype,
        }

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.include_radiomics = include_radiomics
        self.verbose = verbose

    def setup(self, stage=None):
        self.data = np.load(self.data_dir, allow_pickle=True).item()
        self.images = torch.from_numpy(self.data["images"])
        self.slice_idx = self.data.get("slice_idx", None)

        if self.include_radiomics:
            self.radiomics = np.load("./data/radiomics.npy", allow_pickle=True).item()

            for key in self.radiomics.keys():
                self.radiomics[key] = torch.nn.functional.one_hot(
                    torch.from_numpy(self.radiomics[key]).long(),
                    num_classes=2 if key in ["x", "y", "z"] else 3,
                ).float()

            self.radiomics = torch.cat(
                [self.radiomics[key] for key in self.radiomics.keys()], dim=-1
            )

            train_images, train_y, val_images, val_y = (
                train_test_split(
                    self.data,
                    self.radiomics,
                    train_size=self.train_ratio,
                    random_state=42,
                    shuffle=False,
                )
                if self.train_ratio < 1
                else (self.images, self.radiomics, [], [])
            )

            self.train_dataset = BRATSDataset(
                train_images, train_y, **self.dataset_kwargs
            )
            self.val_dataset = BRATSDataset(val_images, val_y)

        else:
            if self.slice_idx is not None:
                self.slice_idx = torch.from_numpy(self.slice_idx)

                # train_images, train_slice_idx, val_images,  val_slice_idx
                train_images, val_images, train_slice_idx, val_slice_idx = (
                    train_test_split(
                        self.images,
                        self.slice_idx,
                        train_size=self.train_ratio,
                        random_state=42,
                        shuffle=False,
                    )
                    if self.train_ratio < 1
                    else (self.images, self.slice_idx, [], [])
                )

                self.train_dataset = BRATSDataset(
                    train_images, train_slice_idx, **self.dataset_kwargs
                )
                self.val_dataset = BRATSDataset(val_images, val_slice_idx)
            else:
                train_images, val_images = (
                    train_test_split(
                        self.images,
                        train_size=self.train_ratio,
                        random_state=42,
                        shuffle=False,
                    )
                    if self.train_ratio < 1
                    else (self.images, [])
                )

                self.train_dataset = BRATSDataset(train_images, **self.dataset_kwargs)
                self.val_dataset = BRATSDataset(val_images)

        log = """
        DataModule setup complete.
        Number of training samples: {}
        Number of validation samples: {}
        Data shape: {}
        Maximum: {}, Minimum: {}
        """.format(
            len(self.train_dataset),
            len(self.val_dataset),
            self.images.shape,
            self.images.max(),
            self.images.min(),
        )

        if self.verbose:
            print(log)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class TumorSRDataset(BRATSDataset):
    def __init__(
        self,
        *data: Tuple[torch.Tensor, ...],
        autoencoder: AutoencoderKL = None,
        window_size: Tuple[int, int] = (64, 64),
        transform=None,
        resize=None,
        horizontal_flip=None,
        vertical_flip=None,
        random_crop_size=None,
        rotation=None,
        dtype=torch.float32
    ):
        super().__init__(
            *data,
            transform=transform,
            resize=resize,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            random_crop_size=random_crop_size,
            rotation=rotation,
            dtype=dtype
        )
        self.window_size = window_size
        self.autoencoder = autoencoder.to("cuda") if autoencoder is not None else None

    def _extract_hr_lr_tumor(self, high_res, **kwargs):
        # Extract mri squences and tumor mask
        tumor_mask = high_res[-1, None, ...]

        if tumor_mask.min() < 0:
            tumor_mask = (tumor_mask > 0).float()

        if torch.sum(tumor_mask) == 0:
            raise ValueError("Tumor mask is empty")

        if self.autoencoder is not None:
            position = kwargs.get("position", None)
            if position is not None:
                low_res = (
                    self.autoencoder.forward(
                        high_res.unsqueeze(0).to(
                            self.autoencoder.device, dtype=self.autoencoder.dtype
                        ),
                        position.to(
                            self.autoencoder.device, dtype=self.autoencoder.dtype
                        ),
                        sample_posterior=True,
                    )
                    .detach()
                    .cpu()
                    .squeeze(0)
                )
            else:
                raise ValueError(
                    "Position is required to autoencode low resolution sequence, otherwise set autoencoder to None"
                )

        high_res = (high_res.add(1).div(2) * tumor_mask).mul(2).sub(1)
        low_res = (low_res.add(1).div(2) * tumor_mask).mul(2).sub(1)

        # Find the bounding box of the tumor mask
        nonzero_indices = torch.nonzero(tumor_mask[0])
        min_x = nonzero_indices[:, 0].min()
        max_x = nonzero_indices[:, 0].max()
        min_y = nonzero_indices[:, 1].min()
        max_y = nonzero_indices[:, 1].max()

        # Calculate center coordinates
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Calculate offsets to extract a 64x64 patch around the tumor center
        offset_x = max(0, center_x - self.window_size[0] // 2)
        offset_y = max(0, center_y - self.window_size[1] // 2)

        # Extract patches around the tumor center from flair and t1ce images
        hs_tumor_patch = high_res[
            :-1,  # removing the tumor mask
            offset_x : offset_x + self.window_size[0],
            offset_y : offset_y + self.window_size[1],
        ]

        ls_tumor_patch = low_res[
            :-1,  # removing the tumor mask
            offset_x : offset_x + self.window_size[0],
            offset_y : offset_y + self.window_size[1],
        ]

        return hs_tumor_patch, ls_tumor_patch, tumor_mask

    def __getitem__(self, index):
        instance = [d[index] for d in self.data]

        sequence = self.transform(instance[0])
        position = instance[1] if len(instance) > 1 else None

        hs_tumor_patch, ls_tumor_patch, tumor_mask = self._extract_hr_lr_tumor(
            sequence, position=position
        )

        return hs_tumor_patch, ls_tumor_patch, tumor_mask


class TumorSRDataModule(BRATSDataModule):
    def __init__(
        self,
        autoencoder: AutoencoderKL,
        data_dir: str,  # should target the a npy file generated via create_*.py scripts
        window_size: Tuple[int, int] = (64, 64),
        train_ratio: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        horizontal_flip=None,
        vertical_flip=None,
        rotation=None,
        random_crop_size=None,
        dtype=torch.float32,
        verbose=True,
        **kwargs
    ):
        super().__init__(
            data_dir,
            train_ratio,
            batch_size,
            num_workers,
            shuffle,
            horizontal_flip,
            vertical_flip,
            rotation,
            random_crop_size,
            dtype,
            verbose,
            False,
            **kwargs
        )
        self.window_size = window_size
        self.autoencoder = autoencoder
        if self.autoencoder is not None:
            self.autoencoder.eval()
            self.autoencoder.requires_grad_(False)

    def setup(self, stage: str = "train") -> None:
        self.data = np.load(self.data_dir, allow_pickle=True).item()
        self.images = torch.from_numpy(self.data["images"])
        self.slice_idx = self.data.get("slice_idx", None)
        self.slice_idx = torch.from_numpy(self.slice_idx)

        train_images, val_images, train_slice_idx, val_slice_idx = (
            train_test_split(
                self.images,
                self.slice_idx,
                train_size=self.train_ratio,
                random_state=42,
                shuffle=False,
            )
            if self.train_ratio < 1
            else (self.images, self.slice_idx, [], [])
        )

        self.train_dataset = TumorSRDataset(
            train_images,
            train_slice_idx,
            autoencoder=self.autoencoder,
            window_size=self.window_size,
            **self.dataset_kwargs
        )

        self.val_dataset = TumorSRDataset(
            val_images,
            val_slice_idx,
            autoencoder=self.autoencoder,
            window_size=self.window_size,
        )

        log = """
        DataModule setup complete.
        Number of training samples: {}
        Number of validation samples: {}
        Data shape: {}
        Maximum: {}, Minimum: {}
        """.format(
            len(self.train_dataset),
            len(self.val_dataset),
            self.images.shape,
            self.images.max(),
            self.images.min(),
        )

        if self.verbose:
            print(log)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *data,
        transform=None,
        resize=None,
        horizontal_flip=None,
        vertical_flip=None,
        random_crop_size=None,
        rotation=None,
        dtype=torch.float32
    ):
        super().__init__()
        self.data = data

        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(resize) if resize is not None else nn.Identity(),
                    (
                        T.RandomHorizontalFlip(p=horizontal_flip)
                        if horizontal_flip
                        else nn.Identity()
                    ),
                    (
                        T.RandomVerticalFlip(p=vertical_flip)
                        if vertical_flip
                        else nn.Identity()
                    ),
                    (
                        T.RandomCrop(random_crop_size)
                        if random_crop_size is not None
                        else nn.Identity()
                    ),
                    (
                        T.RandomRotation(rotation, fill=-1)
                        if rotation is not None
                        else nn.Identity()
                    ),
                    T.ConvertImageDtype(dtype),
                    # WARNING: mean and std are not the target values but rather the values to subtract and divide by:
                    # [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [self.transform(d[index]) for d in self.data]

    def sample(self, n, transform=None):
        """sampling randomly n samples from the dataset, apply or not the transform"""
        transform = self.transform if transform is not None else lambda x: x
        idx = np.random.choice(len(self), n)
        return [
            transform(d[idx]) if i == 0 else d[idx] for i, d in enumerate(self.data)
        ]


class SegAugDataModule(LightningDataModule):
    def __init__(
        self,
        real_dir: str,  # should target the a npy file
        fake_dir: str,
        classes: List[int],
        n_reals: int,
        n_fakes: int,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        horizontal_flip=None,
        vertical_flip=None,
        rotation=None,
        random_crop_size=None,
        dtype="float32",
        verbose=True,
        **kwargs
    ):
        super().__init__()
        self.dataset_kwargs = {
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "random_crop_size": random_crop_size,
            "rotation": rotation,
            "dtype": parse_dtype(dtype),
        }

        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.classes = classes
        self.n_reals = n_reals
        self.n_fakes = n_fakes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.verbose = verbose

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str = None) -> None:
        self.train_real = np.load(self.real_dir, allow_pickle=True).item()["images"]
        # selecting right labels
        self.train_real[:, -1] = np.where(
            np.isin(self.train_real[:, -1], self.classes), 1, 0
        )
        self.train_real, self.test = (
            self.train_real[: self.n_reals],
            self.train_real[self.n_reals + 100 :],
        )

        if self.fake_dir is not None:
            self.train_fake = np.load(self.fake_dir, allow_pickle=True)
            self.train_fake[:, -1] = np.where(
                np.isin(self.train_fake[:, -1], self.classes), 1, 0
            )

            np.random.shuffle(self.train_fake)
            self.train_fake = self.train_fake[: self.n_fakes]
            self.train = np.concatenate([self.train_real, self.train_fake], axis=0)
        else:
            self.train = self.train_real

        self.train_dataset = Dataset(
            torch.from_numpy(self.train), **self.dataset_kwargs
        )
        self.val_dataset = Dataset(torch.from_numpy(self.test))

        log = """
        SegAugDataModule setup complete.
        Number of training samples: {}
        Number of validation samples: {}
        Train shape: {}
        (real) Maximum: {}, Minimum: {},
        (fake) Maximum: {}, Minimum: {},
        Number of classes (real): {},
        Number of classes (fake): {},
        With selected classes: {},
        N° of real samples: {}
        N° of fake samples: {}
        """.format(
            len(self.train_dataset),
            len(self.val_dataset),
            self.train.shape,
            self.train_real[:, :-1].max() if self.n_reals > 0 else None,
            self.train_real[:, :-1].min() if self.n_reals > 0 else None,
            self.train_fake[:, :-1].max() if self.n_fakes > 0 else None,
            self.train_fake[:, :-1].min() if self.n_fakes > 0 else None,
            np.unique(self.train_real[:, -1]),
            np.unique(self.train_fake[:, -1]),
            self.classes,
            self.n_reals,
            self.n_fakes,
        )

        if self.verbose:
            print(log)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
