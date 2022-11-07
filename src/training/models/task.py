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

        self.val_ndcg = RetrievalNormalizedDCG()

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, targets = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets)
        return loss, targets, targets_pred, users

    # def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
    def training_step(self, batch: Any, batch_idx: int):
        loss, targets, targets_pred, _ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_rmse(targets_pred, targets)
        self.log(
            "train/rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=False
        )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets, targets_pred, users = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_rmse(targets_pred, targets)
        self.log(
            "val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=False
        )
        return {
            "loss": loss,
            "targets": targets,
            "targets_pred": targets_pred,
            "users": users,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        targets_pred = torch.cat(([o["targets_pred"] for o in outputs]))
        targets = torch.cat(([o["targets"] for o in outputs]))
        indexes = torch.cat(([o["users"] for o in outputs]))
        self.val_ndcg(targets_pred, targets, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer
        # optimizer1 = torch.optim.SparseAdam(list(self.parameters())[:2], lr=5e-2)
        # optimizer2 = torch.optim.Adam(list(self.parameters())[2:],
        #                               lr=1e-3, weight_decay=1e-4)
        # return optimizer1, optimizer2


class ClassificationTask(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.val_ndcg = RetrievalNormalizedDCG()

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, targets = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets)
        return loss, targets, targets_pred, users

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, _ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets, targets_pred, users = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "targets": targets,
            "targets_pred": targets_pred,
            "users": users,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        targets_pred = torch.cat(([o["targets_pred"] for o in outputs]))
        targets = torch.cat(([o["targets"] for o in outputs]))
        indexes = torch.cat(([o["users"] for o in outputs]))
        self.val_ndcg(targets_pred, targets, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer


def bpr_loss(positive_sim: torch.Tensor, negative_sim: torch.Tensor) -> torch.Tensor:
    distance = positive_sim - negative_sim
    # Probability of ranking given parameters
    elementwise_bpr_loss = torch.log(torch.sigmoid(distance))

    # The goal is to minimize loss
    # If negative sim > positive sim -> distance is negative,
    # but loss is positive
    bpr_loss = -elementwise_bpr_loss.mean()

    return bpr_loss


class BPRTask(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        # sigmoid ?
        self.criterion = bpr_loss

        self.val_ndcg = RetrievalNormalizedDCG()

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def training_step(self, batch: Any, batch_idx: int):
        users, items_neg, items_pos = batch
        pred_neg = self.forward(users, items_neg)
        pred_pos = self.forward(users, items_pos)
        loss = self.criterion(pred_pos, pred_neg)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        users, items, targets = batch
        targets_pred = self.forward(users, items)
        return {
            "targets": targets,
            "targets_pred": targets_pred,
            "users": users,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat(([o["targets_pred"] for o in outputs]))
        target = torch.cat(([o["targets"] for o in outputs]))
        indexes = torch.cat(([o["users"] for o in outputs]))
        self.val_ndcg(preds, target, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer
