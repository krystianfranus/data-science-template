import logging
import os

import hydra
import pandas as pd
import torch
from clearml import Logger, Task, TaskTypes
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mypackage.training.datamodules.dataset import InferDataset
from mypackage.training.models.task import (
    BPRMFTask,
    BPRMLPTask,
    SimpleMFTask,
    SimpleMLPTask,
)

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs", "inference"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    task = Task.init(
        project_name="MyProject",
        task_name="Inference",
        task_type=TaskTypes.inference,
        reuse_last_task_id=False,
        output_uri=None,
    )

    if cfg.execute_remotely:
        task.execute_remotely()

    if cfg.prev_task_id is not None:
        task_prev = Task.get_task(task_id=cfg.prev_task_id)
    else:
        task_prev = Task.get_task(project_name="MyProject", task_name="Training")

    ckpt_path = task_prev.models["input"][-1].get_local_copy()
    match cfg.model_type:
        case "simple_mlp":
            model = SimpleMLPTask.load_from_checkpoint(ckpt_path)
        case "simple_mf":
            model = SimpleMFTask.load_from_checkpoint(ckpt_path)
        case "bpr_mlp":
            model = BPRMLPTask.load_from_checkpoint(ckpt_path)
        case "bpr_mf":
            model = BPRMFTask.load_from_checkpoint(ckpt_path)
        case _:
            raise ValueError(f"Invalid model type, you provided '{cfg.model_type}'")

    n_users = model.net.embed_user.num_embeddings
    n_items = model.net.embed_item.num_embeddings
    dataset = InferDataset(n_users, n_items)
    dataloader = DataLoader(
        dataset,
        batch_size=n_items,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    predictions = Trainer().predict(model, dataloaders=dataloader)
    predictions = torch.concat(predictions).view(n_users, n_items)

    recs = pd.DataFrame(
        predictions.sort(descending=True)[1][:20, :10],
        columns=[f"top{i} item" for i in range(1, 11)],
    )
    Logger.current_logger().report_table(
        "Recommendations for first 20 users",
        "Top 10 items",
        iteration=0,
        table_plot=recs,
    )

    log.info("Done!")


if __name__ == "__main__":
    main()
