import logging

import hydra
import pandas as pd
import torch
from clearml import Logger, Task, TaskTypes
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mypackage import get_project_root
from mypackage.training.datamodules.dataset import InferDataset
from mypackage.training.models.task import (
    BPRMFTask,
    BPRMLPTask,
    SimpleMFTask,
    SimpleMLPTask,
)

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "inference"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    task_training = Task.get_task(
        task_id=cfg.training_task_id,
        project_name=cfg.project_name,
        task_name="Training",
    )

    Task.init(
        project_name=cfg.project_name,
        task_name="Inference",
        task_type=TaskTypes.inference,
        reuse_last_task_id=False,
        output_uri=task_training.output_uri,
    )

    log.info("Loading model")
    ckpt_path = task_training.models["output"][-1].get_local_copy()
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
    log.info("Loading model - success!")

    log.info("Instantiating dataloader")
    n_users = model.net.embed_user.num_embeddings
    n_items = model.net.embed_item.num_embeddings
    dataset = InferDataset(n_users, n_items)
    dataloader = DataLoader(
        dataset,
        batch_size=n_items,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    log.info("Instantiating dataloader - success!")

    log.info("Predicting")
    predictions = Trainer(logger=False).predict(model, dataloaders=dataloader)
    predictions = torch.concat(predictions).view(n_users, n_items)
    log.info("Predicting - success!")

    log.info("Preparing lists with recommendations")
    recs = pd.DataFrame(
        predictions.sort(descending=True)[1][:100, :10],
        columns=[f"top{i} item" for i in range(1, 11)],
    )
    Logger.current_logger().report_table(
        "Recommendations for first 100 users",
        "Top 10 items",
        iteration=0,
        table_plot=recs,
    )
    log.info("Preparing lists with recommendations - success!")


if __name__ == "__main__":
    main()
