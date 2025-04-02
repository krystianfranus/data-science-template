from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SparseAdam

# from torch.optim.lr_scheduler import StepLR
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

from mypackage.training.models.net import MF, MLP


class SimpleMFTask(LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        user_history_based: bool,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MF(n_users, n_items, embed_size, user_history_based)
        self.criterion = BCEWithLogitsLoss()
        self.val_ndcg = RetrievalNormalizedDCG(empty_target_action="error")
        self.val_auroc = RetrievalAUROC(empty_target_action="error")
        self.val_step_outputs = []

    def forward(self, users: Tensor, items: Tensor):
        return self.net(users, items)

    def predict(self, users: Tensor, items: Tensor):
        return torch.sigmoid(self(users, items))

    def predict_step(self, batch, batch_idx):
        users, items = batch
        return self.predict(users, items)

    def step(self, batch):
        list_ids, users, items, targets_true, user_histories = batch
        if self.hparams.user_history_based:
            targets_pred = self.net(user_histories, items)
        else:
            targets_pred = self.net(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, list_ids

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets_true, targets_pred, list_ids = self.step(batch)
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.val_step_outputs.append((targets_pred, targets_true, list_ids))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred, targets_true, list_ids = map(
            torch.cat, zip(*self.val_step_outputs)
        )

        self.val_ndcg(targets_pred, targets_true, indexes=list_ids)
        self.val_auroc(targets_pred, targets_true, indexes=list_ids)
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
        embed_size: int,
        n_layers: int,
        dropout: float,
        user_history_based: bool,
        lr1: float,
        lr2: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MLP(
            n_users, n_items, embed_size, n_layers, dropout, user_history_based
        )
        # self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([13]))
        self.criterion = BCEWithLogitsLoss()
        self.val_ndcg = RetrievalNormalizedDCG(empty_target_action="error")
        self.val_auroc = RetrievalAUROC(empty_target_action="error")

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
        list_ids, users, items, targets_true, user_histories = batch
        if self.hparams.user_history_based:
            targets_pred = self.net(user_histories, items)
        else:
            targets_pred = self.net(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, list_ids

    def training_step(self, batch, batch_idx):
        optimizer1, optimizer2 = self.optimizers()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss, *_ = self.step(batch)
        self.manual_backward(loss)
        optimizer1.step()
        optimizer2.step()
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=False)

        # if self.trainer.is_last_batch:
        #     scheduler1, scheduler2 = self.lr_schedulers()
        #     scheduler1.step()
        #     scheduler2.step()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets_true, targets_pred, list_ids = self.step(batch)
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.val_step_outputs.append((targets_pred, targets_true, list_ids))

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/lightning/pull/16520
        targets_pred, targets_true, list_ids = map(
            torch.cat, zip(*self.val_step_outputs)
        )

        self.val_ndcg(targets_pred, targets_true, indexes=list_ids)
        self.val_auroc(targets_pred, targets_true, indexes=list_ids)
        self.log("ndcg/val", self.val_ndcg, prog_bar=True)
        self.log("auroc/val", self.val_auroc, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.user_history_based:
            optimizer1 = SparseAdam(list(self.parameters())[:1], lr=self.hparams.lr1)
            optimizer2 = Adam(
                list(self.parameters())[1:],
                lr=self.hparams.lr2,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer1 = SparseAdam(list(self.parameters())[:2], lr=self.hparams.lr1)
            optimizer2 = Adam(
                list(self.parameters())[2:],
                lr=self.hparams.lr2,
                weight_decay=self.hparams.weight_decay,
            )
        # scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.98)
        # scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.99)
        # return [optimizer1, optimizer2], [scheduler1, scheduler2]
        return optimizer1, optimizer2
