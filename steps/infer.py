import logging
import os

import hydra
import pandas as pd
import torch
from clearml import Logger, Task, TaskTypes
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

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
    n_users = int(task_prev.get_parameter("General/n_users"))
    n_items = int(task_prev.get_parameter("General/n_items"))

    log.info("Instantiating model")
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

    log.info("Preparing dataloader")

    class TmpDataset(Dataset):
        def __init__(self, n_users, n_items):
            self.n_users = n_users
            self.n_items = n_items

        def __len__(self):
            return self.n_users * self.n_items

        def __getitem__(self, idx):
            user = idx // self.n_users
            item = idx % self.n_items
            return user, item

    dataset = TmpDataset(n_users, n_items)
    dataloader = DataLoader(dataset, batch_size=n_items, num_workers=8, pin_memory=True)

    log.info("Main loop")
    model.eval()
    device = torch.device("cuda:0")
    big_scores = torch.empty((n_users, n_items))
    with torch.no_grad():
        for i, (users, items) in enumerate(dataloader):
            users = users.to(device)
            items = items.to(device)
            scores = model.predict(users, items)
            big_scores[i, :] = scores.cpu()

    # Log top10 recommendations
    tmp = pd.DataFrame(
        big_scores.sort(descending=True)[1][:20, :10],
        columns=[f"top{i} item" for i in range(1, 11)],
    )
    Logger.current_logger().report_table(
        "Recommendations for first 20 users",
        "Top 10",
        iteration=0,
        table_plot=tmp,
    )

    # log.info("[My Logger] Instantiating trainer")
    # trainer = pl.Trainer(
    #     logger=False,
    #     enable_checkpointing=False,
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=5,
    #     log_every_n_steps=5,
    # )
    # log.info("[My Logger] Inferring")
    # predictions = trainer.predict(model, dataloader, ckpt_path=ckpt_path)
    # mean_predictions = torch.mean(torch.concat(predictions))
    # log.info(f"[My Logger] Results - Mean predictions: {mean_predictions}")

    log.info("Done!")


if __name__ == "__main__":
    main()
