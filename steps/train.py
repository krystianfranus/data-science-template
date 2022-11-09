import logging

import hydra
import pandas as pd
from clearml import Task, TaskTypes
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/training/", config_name="config")
def main(config: DictConfig):
    if config.execute_locally:
        log.info("[My Logger] Loading data (artifacts)")
        data_dir = "data/movielens"
        train_data = pd.read_csv(f"{data_dir}/train_data_{config.data_type}.csv")
        val_data = pd.read_csv(f"{data_dir}/val_data_{config.data_type}.csv")
        test_data = pd.read_csv(f"{data_dir}/test_data_{config.data_type}.csv")
        # train_data = f"{data_dir}/train_data_{config.data_type}.csv"  # Iterable
        # val_data = f"{data_dir}/val_data_{config.data_type}.csv"  # Iterable
        # test_data = f"{data_dir}/test_data_{config.data_type}.csv"  # Iterable
    else:
        task = Task.init(
            project_name="My project",
            task_name="Training",
            task_type=TaskTypes.training,
            output_uri="s3://kfranus-bucket/data-science-template/output/",
        )

        if config.prev_task_id is not None:
            task_prev = Task.get_task(task_id=config.prev_task_id)
        else:
            task_prev = Task.get_task(
                project_name="My project", task_name="Preprocessing"
            )

        log.info("[My Logger] Loading data (artifacts)")
        train_data = task_prev.artifacts["train_data"].get()
        val_data = task_prev.artifacts["val_data"].get()
        test_data = task_prev.artifacts["test_data"].get()

        if config.draft_mode:
            task.execute_remotely()

    log.info("[My Logger] Instantiating datamodule")
    datamodule_params = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
    }
    datamodule = hydra.utils.instantiate(config.datamodule, **datamodule_params)

    log.info("[My Logger] Instantiating model")
    net_params = {"n_users": 6040, "n_items": 3706}  # 28597, 6733
    net = hydra.utils.instantiate(config.net, **net_params)
    optimizer = hydra.utils.instantiate(config.optimizer)
    model = hydra.utils.instantiate(config.model, net=net, optimizer=optimizer)

    log.info("[My Logger] Instantiating callbacks")
    callbacks = []
    for _, cb_cfg in config.callbacks.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    log.info("[My Logger] Instantiating trainer")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    log.info("[My Logger] Training")
    trainer.fit(model=model, datamodule=datamodule)
    # log.info("[My Logger] Testing")
    # trainer.test(model=model, datamodule=datamodule)

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
