from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SparseAdam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAUROC
from torchmetrics.retrieval import RetrievalNormalizedDCG

from mypackage.training.models.net import MF, MLP


class SimpleMFTask(LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        lr: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MF(n_users, n_items, embed_size)
        pos_weight = torch.tensor([kwargs["n_impressions"] / kwargs["n_clicks"]])
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_auroc = BinaryAUROC()
        self.val_step_outputs = []

    def forward(self, users: Tensor, items: Tensor):
        return self.net(users, items)

    def predict(self, users: Tensor, items: Tensor):
        return torch.sigmoid(self(users, items))

    def predict_step(self, batch, batch_idx):
        users, items = batch
        return self.predict(users, items)

    def step(self, batch):
        users, items, targets_true = batch
        targets_pred = self.net(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, users

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred, targets_true, users = map(torch.cat, zip(*self.val_step_outputs))

        self.val_ndcg(targets_pred, targets_true, indexes=users)
        self.val_auroc(targets_pred, targets_true)
        self.log("ndcg/val", self.val_ndcg, prog_bar=True)
        self.log("auroc/val", self.val_auroc, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = SparseAdam(params=self.parameters(), lr=self.hparams.lr)
        return optimizer


class SimpleMLPTask(LightningModule):
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MLP(n_users, n_items, n_factors, n_layers, dropout)
        pos_weight = torch.tensor([kwargs["n_impressions"] / kwargs["n_clicks"]])
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_auroc = BinaryAUROC()

        self.val_step_outputs = []
        self.automatic_optimization = False

    def forward(self, users: Tensor, items: Tensor):
        return self.net(users, items)

    def predict(self, users: Tensor, items: Tensor):
        return torch.sigmoid(self(users, items))

    def predict_step(self, batch, batch_idx):
        users, items = batch
        return self.predict(users, items)

    def step(self, batch: Any):
        users, items, targets_true = batch
        targets_pred = self.net(users, items)
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
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss, *_ = self.step(batch)
        self.manual_backward(loss)
        optimizer1.step()
        optimizer2.step()
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.trainer.is_last_batch:
            scheduler1, scheduler2 = self.lr_schedulers()
            scheduler1.step()
            scheduler2.step()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets_true, targets_pred, users = self.step(batch)
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred, targets_true, users = map(torch.cat, zip(*self.val_step_outputs))
        self.val_ndcg(targets_pred, targets_true, indexes=users)
        self.val_auroc(targets_pred, targets_true)
        self.log("ndcg/val", self.val_ndcg, prog_bar=True)
        self.log("auroc/val", self.val_auroc, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer1 = SparseAdam(list(self.parameters())[:2], lr=self.hparams.lr1)
        optimizer2 = Adam(
            list(self.parameters())[2:],
            lr=self.hparams.lr2,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.98)
        scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.99)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]


def bpr_loss(positive_sim: Tensor, negative_sim: Tensor) -> Tensor:
    distance = positive_sim - negative_sim
    # Probability of ranking given parameters
    elementwise_bpr_loss = torch.log(torch.sigmoid(distance))

    # The goal is to minimize loss
    # If negative sim > positive sim -> distance is negative,
    # but loss is positive
    bpr_loss = -elementwise_bpr_loss.mean()

    return bpr_loss


class BPRMFTask(LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        lr: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MF(n_users, n_items, embed_size)
        # sigmoid ?
        self.criterion = bpr_loss
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_auroc = BinaryAUROC()
        self.val_step_outputs = []

    def forward(self, users: Tensor, items: Tensor):
        return self.net(users, items)

    def predict(self, users: Tensor, items: Tensor):
        return torch.sigmoid(self(users, items))

    def predict_step(self, batch, batch_idx):
        users, items = batch
        return self.predict(users, items)

    def training_step(self, batch, batch_idx):
        users, items_neg, items_pos = batch
        pred_neg = self.net(users, items_neg)
        pred_pos = self.net(users, items_pos)
        loss = self.criterion(pred_pos, pred_neg)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        loss = self.criterion(
            targets_pred, targets_true
        )  # TODO: THIS IS INCORRECT LOSS COMPUTATION
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred, targets_true, users = map(torch.cat, zip(*self.val_step_outputs))
        self.val_ndcg(targets_pred, targets_true, indexes=users)
        self.val_auroc(targets_pred, targets_true)
        self.log("ndcg/val", self.val_ndcg, prog_bar=True)
        self.log("auroc/val", self.val_auroc, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = SparseAdam(params=self.parameters(), lr=self.hparams.lr)
        return optimizer


class BPRMLPTask(LightningModule):
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MLP(n_users, n_items, n_factors, n_layers, dropout)
        # sigmoid ?
        self.criterion = bpr_loss
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_auroc = BinaryAUROC()

        self.val_step_outputs = []
        self.automatic_optimization = False

    def forward(self, users: Tensor, items: Tensor):
        return self.net(users, items)

    def predict(self, users: Tensor, items: Tensor):
        return torch.sigmoid(self(users, items))

    def predict_step(self, batch, batch_idx):
        users, items = batch
        return self.predict(users, items)

    def training_step(self, batch, batch_idx):
        optimizer1, optimizer2 = self.optimizers()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        users, items_neg, items_pos = batch
        pred_neg = self.net(users, items_neg)
        pred_pos = self.net(users, items_pos)
        loss = self.criterion(pred_pos, pred_neg)
        self.manual_backward(loss)
        optimizer1.step()
        optimizer2.step()
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
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

        if self.trainer.is_last_batch:
            scheduler1, scheduler2 = self.lr_schedulers()
            scheduler1.step()
            scheduler2.step()

        return loss

    def validation_step(self, batch, batch_idx):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(
            targets_pred, targets_true
        )  # TODO: THIS IS INCORRECT LOSS COMPUTATION
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append((targets_pred, targets_true, users))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred, targets_true, users = map(torch.cat, zip(*self.val_step_outputs))
        self.val_ndcg(targets_pred, targets_true, indexes=users)
        self.val_auroc(targets_pred, targets_true)
        self.log("ndcg/val", self.val_ndcg, prog_bar=True)
        self.log("auroc/val", self.val_auroc, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer1 = SparseAdam(list(self.parameters())[:2], lr=self.hparams.lr1)
        optimizer2 = Adam(
            list(self.parameters())[2:],
            lr=self.hparams.lr2,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.95)
        scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.97)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]
