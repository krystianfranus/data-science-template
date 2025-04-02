import logging

import hydra
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
        tags=cfg.tags,
        reuse_last_task_id=False,
    )

    log.info("Loading data")
    train = task_data_processing.artifacts["train"].get()
    val = task_data_processing.artifacts["validation"].get()
    user_mapper = task_data_processing.artifacts["user_mapper"].get()
    item_mapper = task_data_processing.artifacts["item_mapper"].get()
    log.info("Data loaded successfully!")

    log.info("Instantiating datamodule")
    datamodule_params = {
        "train": train,
        "val": val,
    }
    datamodule = hydra.utils.instantiate(cfg.datamodule, **datamodule_params)
    log.info("Datamodule instantiated successfully!")

    log.info("Instantiating model")
    model_params = {
        "n_users": len(user_mapper),
        "n_items": len(item_mapper),
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
