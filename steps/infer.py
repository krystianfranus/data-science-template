import logging

import hydra
import pytorch_lightning as pl
import torch
from clearml import Task, TaskTypes
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.training.datamodules.dataset import PredictDataset

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/inference/", config_name="config"
)
def main(config: DictConfig):
    task = Task.init(
        project_name="My project",
        task_name="Inference",
        task_type=TaskTypes.inference,
    )

    if config.execute_remotely:
        task.execute_remotely(queue_name="default")

    if config.prev_task_id is not None:
        task_prev = Task.get_task(task_id=config.prev_task_id)
    else:
        task_prev = Task.get_task(project_name="My project", task_name="Training")

    log.info("[My Logger] Preparing dataloader")
    x = torch.tensor([[1.0], [1.5]])
    dataset = PredictDataset(x)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    log.info("[My Logger] Instantiating model")
    model = hydra.utils.instantiate(config.model)
    ckpt_path = task_prev.models["output"][-1].get_local_copy()

    log.info("[My Logger] Instantiating trainer")
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        log_every_n_steps=5,
    )

    log.info("[My Logger] Inferring")
    predictions = trainer.predict(model, dataloader, ckpt_path=ckpt_path)

    mean_predictions = torch.mean(torch.concat(predictions))
    log.info(f"[My Logger] Results - Mean predictions: {mean_predictions}")

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
