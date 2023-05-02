from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn, optim
from torchmetrics.retrieval import RetrievalNormalizedDCG

from mypackage.training.models.net import MF, MLP


class SimpleMFTask(pl.LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MF(n_users, n_items, embed_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_step_outputs = []

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, users

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.logger.experiment.add_histogram(
                "embed_user",
                self.net.embed_user.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
            self.logger.experiment.add_histogram(
                "embed_item",
                self.net.embed_item.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets_true, targets_pred, users = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred = torch.cat([tp for tp, *_ in self.val_step_outputs])
        targets_true = torch.cat([tt for _, tt, _ in self.val_step_outputs])
        indexes = torch.cat([u for *_, u in self.val_step_outputs])
        self.val_ndcg(targets_pred, targets_true, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.SparseAdam(params=self.parameters(), lr=self.hparams.lr)
        return optimizer


class SimpleMLPTask(pl.LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        n_layers: int,
        dropout: float,
        lr1: float,
        lr2: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MLP(n_users, n_items, n_factors, n_layers, dropout)
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_step_outputs = []
        self.automatic_optimization = False

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        return torch.sigmoid(self.net(users, items))

    def step(self, batch: Any):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, users

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.logger.experiment.add_histogram(
                "embed_user",
                self.net.embed_user.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
            self.logger.experiment.add_histogram(
                "embed_item",
                self.net.embed_item.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )

        optimizer1, optimizer2 = self.optimizers()
        optimizer2.zero_grad()
        optimizer2.zero_grad()
        loss, *_ = self.step(batch)
        self.manual_backward(loss)
        optimizer1.step()
        optimizer2.step()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets_true, targets_pred, users = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred = torch.cat([tp for tp, *_ in self.val_step_outputs])
        targets_true = torch.cat([tt for _, tt, _ in self.val_step_outputs])
        indexes = torch.cat([u for *_, u in self.val_step_outputs])
        self.val_ndcg(targets_pred, targets_true, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer1 = optim.SparseAdam(list(self.parameters())[:2], lr=self.hparams.lr1)
        optimizer2 = optim.Adam(
            list(self.parameters())[2:],
            lr=self.hparams.lr2,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer1, optimizer2


def bpr_loss(positive_sim: torch.Tensor, negative_sim: torch.Tensor) -> torch.Tensor:
    distance = positive_sim - negative_sim
    # Probability of ranking given parameters
    elementwise_bpr_loss = torch.log(torch.sigmoid(distance))

    # The goal is to minimize loss
    # If negative sim > positive sim -> distance is negative,
    # but loss is positive
    bpr_loss = -elementwise_bpr_loss.mean()

    return bpr_loss


class BPRMFTask(pl.LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MF(n_users, n_items, embed_size)
        # sigmoid ?
        self.criterion = bpr_loss
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_step_outputs = []

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def training_step(self, batch, batch_idx):
        users, items_neg, items_pos = batch
        pred_neg = self.forward(users, items_neg)
        pred_pos = self.forward(users, items_pos)
        loss = self.criterion(pred_pos, pred_neg)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.logger.experiment.add_histogram(
                "embed_user",
                self.net.embed_user.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
            self.logger.experiment.add_histogram(
                "embed_item",
                self.net.embed_item.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
        return loss

    def validation_step(self, batch, batch_idx):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets_true)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred = torch.cat([tp for tp, *_ in self.val_step_outputs])
        targets_true = torch.cat([tt for _, tt, _ in self.val_step_outputs])
        indexes = torch.cat([u for *_, u in self.val_step_outputs])
        self.val_ndcg(targets_pred, targets_true, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.SparseAdam(params=self.parameters(), lr=self.hparams.lr)
        return optimizer


class BPRMLPTask(pl.LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        n_layers: int,
        dropout: float,
        lr1: float,
        lr2: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MLP(n_users, n_items, n_factors, n_layers, dropout)
        # sigmoid ?
        self.criterion = bpr_loss
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_step_outputs = []
        self.automatic_optimization = False

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def training_step(self, batch, batch_idx):
        optimizer1, optimizer2 = self.optimizers()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        users, items_neg, items_pos = batch
        pred_neg = self.forward(users, items_neg)
        pred_pos = self.forward(users, items_pos)
        loss = self.criterion(pred_pos, pred_neg)
        self.manual_backward(loss)
        optimizer1.step()
        optimizer2.step()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.logger.experiment.add_histogram(
                "embed_user",
                self.net.embed_user.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
            self.logger.experiment.add_histogram(
                "embed_item",
                self.net.embed_item.weight.data[:, 0],
                self.current_epoch,
                bins="fd",
            )
        return loss

    def validation_step(self, batch, batch_idx):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets_true)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred = torch.cat([tp for tp, *_ in self.val_step_outputs])
        targets_true = torch.cat([tt for _, tt, _ in self.val_step_outputs])
        indexes = torch.cat([u for *_, u in self.val_step_outputs])
        self.val_ndcg(targets_pred, targets_true, indexes=indexes)
        self.log("val/ndcg", self.val_ndcg, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer1 = optim.SparseAdam(list(self.parameters())[:2], lr=self.hparams.lr1)
        optimizer2 = optim.Adam(
            list(self.parameters())[2:],
            lr=self.hparams.lr2,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer1, optimizer2
