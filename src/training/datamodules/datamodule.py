from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.training.datamodules.dataset import SimpleDataset


class SimpleDataModule(LightningDataModule):
    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = SimpleDataset(self.train_data)
            self.val_dataset = SimpleDataset(self.val_data)

        if stage == "test":
            self.test_dataset = SimpleDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )
