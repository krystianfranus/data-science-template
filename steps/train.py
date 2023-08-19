import logging
import os

import hydra
import torch
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(get_project_root(), "configs", "training"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    if cfg.use_remote_storage:
        output_uri = "s3://kf-north-bucket/data-science-template/output/"
    else:
        output_uri = None
    task = Task.init(
        project_name="MyProject",
        task_name="Training",
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=output_uri,
    )

    if cfg.prev_task_id:
        task_prev = Task.get_task(task_id=cfg.prev_task_id)
    else:
        task_prev = Task.get_task(project_name="MyProject", task_name="Preprocessing")
    train = task_prev.artifacts["train"].get()
    val = task_prev.artifacts["val"].get()
    test = task_prev.artifacts["test"].get()
    # if cfg.draft_mode:
    #     task.execute_remotely()

    log.info("Instantiating datamodule")
    datamodule_params = {"train": train, "val": val, "test": test}
    datamodule = hydra.utils.instantiate(cfg.datamodule, **datamodule_params)

    log.info("Instantiating model")
    params = {}
    for k, v in task_prev.get_parameters().items():
        if "General" in k:
            k = k.replace("General/", "")
            params[k] = int(v)
    model = hydra.utils.instantiate(cfg.model, **params)
    task.connect(params)

    log.info("Instantiating callbacks")
    callbacks = []
    for _, cb_cfg in cfg.callbacks.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    log.info("Instantiating trainer")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, default_root_dir=get_project_root()
    )

    log.info("Training")
    trainer.fit(model=model, datamodule=datamodule)

    log.info("Done!")


if __name__ == "__main__":
    main()
