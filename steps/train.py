import logging

import hydra
import numpy as np
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig
from torchinfo import summary

from mypackage import get_project_root

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "training"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    task_data_processing = Task.get_task(
        task_id=cfg.data_processing_task_id,
        project_name=cfg.project_name,
        task_name="DataProcessing",
    )

    Task.init(
        project_name=cfg.project_name,
        task_name="Training",
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=task_data_processing.output_uri,
    )

    log.info("Loading data")
    train = task_data_processing.artifacts["train"].get()
    val = task_data_processing.artifacts["val"].get()
    test = task_data_processing.artifacts["test"].get()
    stats = task_data_processing.artifacts["stats"].get()
    log.info("Data loaded successfully!")

    log.info("Instantiating datamodule")
    batch_size = cfg.datamodule.batch_size
    batch_size = batch_size if batch_size else int(np.sqrt(len(train)))
    datamodule_params = {
        "train": train,
        "val": val,
        "test": test,
        "batch_size": batch_size,
    }
    datamodule = hydra.utils.instantiate(cfg.datamodule, **datamodule_params)
    log.info("Datamodule instantiated successfully!")

    log.info("Instantiating model")
    model_params = {
        "n_users": stats["train_n_users"],
        "n_items": stats["train_n_items"],
    }
    model = hydra.utils.instantiate(cfg.model, **model_params)
    summary(model.net)
    log.info("Model instantiated successfully!")

    log.info("Instantiating callbacks")
    callbacks = []
    for _, cb_cfg in cfg.callbacks.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))
    log.info("Callbacks instantiated successfully!")

    log.info("Instantiating trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    log.info("Trainer instantiated successfully!")

    log.info("Training")
    trainer.fit(model=model, datamodule=datamodule)
    log.info("Model trained successfully!")


if __name__ == "__main__":
    main()
