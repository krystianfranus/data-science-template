from typing import Optional

import lightning.pytorch as pl
import numpy as np
from pandas import DataFrame
from torch.utils.data import DataLoader

from mypackage.training.datamodules.dataset import SimpleDataset


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DataFrame,
        val: DataFrame,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        # self.save_hyperparameters(logger=False)
        self.train = train
        self.val = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if batch_size is None:
            self.batch_size = int(np.sqrt(len(self.train)))

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = SimpleDataset(self.train)
            self.val_dataset = SimpleDataset(self.val)
            # self.val_dataset = ListGroupedDataset(self.val)

        if stage == "test":
            pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        # return DataLoader(
        #     dataset=self.val_dataset,
        #     batch_sampler=ListBatchSampler(self.val_dataset),
        #     collate_fn=collate_fn,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        # )
