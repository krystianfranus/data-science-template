from typing import Any, List

import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError, RetrievalNormalizedDCG


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

        self.val_ndcg = RetrievalNormalizedDCG()

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, ratings = batch
        ratings_pred = self.forward(users, items)
        loss = self.criterion(ratings_pred, ratings)
        return loss, ratings, ratings_pred, users

    def training_step(self, batch: Any, batch_idx: int):
        loss, ratings, ratings_pred, _ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_rmse(ratings_pred, ratings)
        self.log(
            "train/rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=False
        )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, ratings, ratings_pred, users = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_rmse(ratings_pred, ratings)
        self.log(
            "val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=False
        )
        return {
            "loss": loss,
            "ratings": ratings,
            "ratings_pred": ratings_pred,
            "users": users,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat(([o["ratings_pred"] for o in outputs]))
        target = torch.cat(([o["ratings"] for o in outputs]))
        indexes = torch.cat(([o["users"] for o in outputs]))
        self.val_ndcg(preds, target, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, ratings, ratings_pred, _ = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_rmse(ratings_pred, ratings)
        self.log(
            "test/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=False
        )
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer
