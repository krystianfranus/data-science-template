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
    output_uri = None
    if cfg.use_remote_storage:
        output_uri = "s3://kf-north-bucket/data-science-template/output/"

    task = Task.init(
        project_name="MyProject",
        task_name="Training",
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=output_uri,
    )
    if cfg.draft_mode:
        task.execute_remotely()

    task_prev = Task.get_task(
        task_id=cfg.prev_task_id,
        project_name="MyProject",
        task_name="DataProcessing",
    )

    log.info("Loading data")
    train = task_prev.artifacts["train"].get()
    val = task_prev.artifacts["val"].get()
    test = task_prev.artifacts["test"].get()
    log.info("Loading data - success!")

    log.info("Instantiating datamodule")
    datamodule_params = {"train": train, "val": val, "test": test}
    datamodule = hydra.utils.instantiate(cfg.datamodule, **datamodule_params)
    log.info("Instantiating datamodule - success!")

    log.info("Instantiating model")
    n_users = train["user"].nunique()
    n_items = train["item"].nunique()
    model = hydra.utils.instantiate(cfg.model, n_users=n_users, n_items=n_items)
    summary(model.net)
    log.info("Instantiating model - success!")

    log.info("Instantiating callbacks")
    callbacks = []
    for _, cb_cfg in cfg.callbacks.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))
    log.info("Instantiating callbacks - success!")

    log.info("Instantiating trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    log.info("Instantiating trainer - success!")

    log.info("Training")
    trainer.fit(model=model, datamodule=datamodule)
    log.info("Training - success!")


if __name__ == "__main__":
    main()
