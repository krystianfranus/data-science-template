from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import MeanMetric


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

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, ratings = batch
        ratings_pred = self.forward(users, items)
        loss = self.criterion(ratings_pred, ratings)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer
