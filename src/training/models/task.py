from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError


class RegressionTask(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = torch.nn.MSELoss()

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, ratings = batch
        ratings_pred = self.forward(users, items)
        loss = self.criterion(ratings_pred, ratings)
        return loss, ratings, ratings_pred

    def training_step(self, batch: Any, batch_idx: int):
        loss, ratings, ratings_pred = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_rmse(ratings_pred, ratings)
        self.log(
            "train/rmse", self.train_rmse, on_step=True, on_epoch=True, prog_bar=True
        )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, ratings, ratings_pred = self.step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_rmse(ratings_pred, ratings)
        self.log("val/rmse", self.val_rmse, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, ratings, ratings_pred = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.test_rmse(ratings_pred, ratings)
        self.log(
            "test/rmse", self.test_rmse, on_step=True, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer
