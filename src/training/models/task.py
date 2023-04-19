from typing import Any, List

import lightning.pytorch as pl
import torch
from torchmetrics.retrieval import RetrievalNormalizedDCG


class ClassificationMFTask(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.optimizer = torch.optim.SparseAdam
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_step_outputs = []

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, users

    def training_step(self, batch: Any, batch_idx: int):
        loss, *_ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.logger.experiment.add_histogram(
                "embed_user", self.net.embed_user.weight, self.current_epoch, bins="fd"
            )
            self.logger.experiment.add_histogram(
                "embed_item", self.net.embed_item.weight, self.current_epoch, bins="fd"
            )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
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
        optimizer = self.optimizer(params=self.parameters(), lr=self.hparams.lr)
        return optimizer


class ClassificationMLPTask(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.val_ndcg = RetrievalNormalizedDCG()
        self.val_step_outputs = []
        self.automatic_optimization = False

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        return self.net(users, items)

    def step(self, batch: Any):
        users, items, targets_true = batch
        targets_pred = self.forward(users, items)
        loss = self.criterion(targets_pred, targets_true)
        return loss, targets_true, targets_pred, users

    def training_step(self, batch: Any, batch_idx: int):
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()
        loss, *_ = self.step(batch)
        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.logger.experiment.add_histogram(
                "embed_user", self.net.embed_user.weight, self.current_epoch, bins="fd"
            )
            self.logger.experiment.add_histogram(
                "embed_item", self.net.embed_item.weight, self.current_epoch, bins="fd"
            )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
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
        optimizer1 = torch.optim.SparseAdam(
            list(self.parameters())[:2], lr=self.hparams.lr
        )
        optimizer2 = torch.optim.Adam(
            list(self.parameters())[2:],
            lr=self.hparams.lr,
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
