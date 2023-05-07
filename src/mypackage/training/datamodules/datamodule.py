from typing import Optional

import lightning.pytorch as pl
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from mypackage.training.datamodules.dataset import BPRDataset, TrainDataset

# from mypackage.training.datamodules.dataset import CustomIterableDataset


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DataFrame,
        val: DataFrame,
        test: DataFrame,
        batch_size: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = TrainDataset(self.hparams.train)
            self.val_dataset = TrainDataset(self.hparams.val)
            # self.train_dataset = CustomIterableDataset(self.hparams.train)
            # self.val_dataset = CustomIterableDataset(self.hparams.val)

        if stage == "test":
            self.test_dataset = TrainDataset(self.hparams.test)
            # self.test_dataset = CustomIterableDataset(self.hparams.test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class BPRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DataFrame,
        val: DataFrame,
        test: DataFrame,
        batch_size: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = BPRDataset(self.hparams.train)
            self.val_dataset = TrainDataset(self.hparams.val)

        if stage == "test":
            self.test_dataset = TrainDataset(self.hparams.test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
