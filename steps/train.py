import logging
from pathlib import Path

import hydra
import pandas as pd
from clearml import Task, TaskTypes
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs/training/",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    if cfg.use_remote_storage:
        output_uri = "s3://kf-north-bucket/data-science-template/output/"
    else:
        output_uri = None

    log.info("Data loading")
    if cfg.clearml:
        task = Task.init(
            project_name="MyProject",
            task_name="Training",
            task_type=TaskTypes.training,
            output_uri=output_uri,
        )

        if cfg.prev_task_id is not None:
            task_prev = Task.get_task(task_id=cfg.prev_task_id)
        else:
            task_prev = Task.get_task(
                project_name="MyProject", task_name="Preprocessing"
            )

        train_data = task_prev.artifacts["train_data"].get()
        val_data = task_prev.artifacts["val_data"].get()
        test_data = task_prev.artifacts["test_data"].get()

        if cfg.draft_mode:
            task.execute_remotely()
    else:
        prefix = Path("data/contentwise/")
        train_data_path = prefix / Path(f"train_data_{cfg.data_type}.parquet")
        val_data_path = prefix / Path(f"val_data_{cfg.data_type}.parquet")
        test_data_path = prefix / Path(f"test_data_{cfg.data_type}.parquet")
        train_data = pd.read_parquet(train_data_path)
        val_data = pd.read_parquet(val_data_path)
        test_data = pd.read_parquet(test_data_path)

    log.info("Datamodule Instantiating")
    datamodule_params = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
    }
    datamodule = hydra.utils.instantiate(cfg.datamodule, **datamodule_params)

    log.info("Model instantiating")
    net_params = {"n_users": 28450, "n_items": 6706}  # contentwise polars
    # net_params = {"n_users": 28028, "n_items": 6706}  # contentwise bpr polars

    net = hydra.utils.instantiate(cfg.net, **net_params)
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    model = hydra.utils.instantiate(cfg.model, net=net, optimizer=optimizer)

    log.info("Callbacks instantiating")
    callbacks = []
    for _, cb_cfg in cfg.callbacks.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    log.info("Trainer instantiating")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    log.info("Training")
    trainer.fit(model=model, datamodule=datamodule)
    # log.info("[My Logger] Testing")
    # trainer.test(model=model, datamodule=datamodule)

    log.info("Done!")


if __name__ == "__main__":
    main()
