import logging

import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/training/", config_name="config")
def main(config: DictConfig):
    Task.init(project_name="ds_template", task_name="training")
    task_prev = Task.get_task(project_name="ds_template", task_name="preprocessing")

    log.info("[My Logger] Loading data (artifacts)")
    train_data = task_prev.artifacts["train_data"].get()
    val_data = task_prev.artifacts["val_data"].get()
    test_data = task_prev.artifacts["test_data"].get()

    log.info("[My Logger] Instantiating datamodule")
    params = {"train_data": train_data, "val_data": val_data, "test_data": test_data}
    datamodule = hydra.utils.instantiate(config.datamodule, **params)

    log.info("[My Logger] Instantiating model")
    model = hydra.utils.instantiate(config.model)

    log.info("[My Logger] Instantiating callbacks")
    callbacks = []
    for _, cb_cfg in config.callbacks.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    log.info("[My Logger] Instantiating trainer")
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        max_epochs=50,
        log_every_n_steps=5,
    )

    log.info("[My Logger] Training")
    trainer.fit(model=model, datamodule=datamodule)
    log.info("[My Logger] Testing")
    trainer.test(model=model, datamodule=datamodule)

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
