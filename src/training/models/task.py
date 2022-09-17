from typing import Any

import pytorch_lightning as pl
import torch

from src.training.models.net import LinearRegressionNet


class RegressionTask(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = LinearRegressionNet()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
