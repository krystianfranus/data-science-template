import logging
import os

import hydra
import pandas as pd
import torch
from clearml import Logger, Task, TaskTypes
from omegaconf import DictConfig
from torchmetrics import RetrievalNormalizedDCG

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs", "baseline"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    Task.init(
        project_name="MyProject",
        task_name="Baseline",
        task_type=TaskTypes.custom,
        reuse_last_task_id=False,
        output_uri=None,
    )

    if cfg.prev_task_id is not None:
        task_prev = Task.get_task(task_id=cfg.prev_task_id)
    else:
        task_prev = Task.get_task(project_name="MyProject", task_name="Preprocessing")

    log.info("Data loading")
    train_data = task_prev.artifacts["train_data"].get()
    val_data = task_prev.artifacts["val_data"].get()
    n_clicks_per_item = (
        train_data.groupby("item")
        .agg({"target": "sum"})
        .rename(columns={"target": "popularity"})
        .reset_index()
    )
    val_data = val_data.merge(n_clicks_per_item, "inner", "item")

    log.info("NDCG computing")
    ndcg = RetrievalNormalizedDCG()

    indexes = torch.tensor(val_data["user"])
    target = torch.tensor(val_data["target"])
    worst_preds = torch.tensor((val_data["target"] + 1) % 2, dtype=torch.float32)
    random_preds = torch.rand(val_data.shape[0])
    top_preds = torch.tensor(val_data["popularity"], dtype=torch.float32)
    best_preds = torch.tensor(val_data["target"], dtype=torch.float32)

    worst_ndcg = ndcg(worst_preds, target, indexes=indexes).item()
    random_ndcg = ndcg(random_preds, target, indexes=indexes).item()
    pop_ndcg = ndcg(top_preds, target, indexes=indexes).item()
    best_ndcg = ndcg(best_preds, target, indexes=indexes).item()

    baseline = pd.DataFrame()
    baseline.loc["Worst model", "value"] = worst_ndcg
    baseline.loc["Random model", "value"] = random_ndcg
    baseline.loc["Popularity-based model", "value"] = pop_ndcg
    baseline.loc["Best model", "value"] = best_ndcg

    # Log NDCG
    Logger.current_logger().report_table(
        "Baselines of validation data",
        "NDCG",
        iteration=0,
        table_plot=baseline,
    )

    # Log items popularity
    tmp = n_clicks_per_item.sort_values("popularity", ascending=False).reset_index(
        drop=True
    )
    Logger.current_logger().report_table(
        "Items popularity (based on clicks)",
        "Bottom 10",
        iteration=0,
        table_plot=tmp[-10:],
    )
    Logger.current_logger().report_table(
        "Items popularity (based on clicks)",
        "Top 10",
        iteration=0,
        table_plot=tmp[:10],
    )

    log.info("Done!")


if __name__ == "__main__":
    main()
