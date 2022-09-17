import logging

import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/training/", config_name="config")
def main(config: DictConfig):
    Task.init(project_name="ds_template", task_name="training")
    task2 = Task.get_task(task_id="bbc9002386f64032be058972530b0ad3")

    # Load data
    train_data = task2.artifacts["train_data"].get()
    val_data = task2.artifacts["val_data"].get()
    test_data = task2.artifacts["test_data"].get()

    # MAIN CODE
    log.info("Training")
    params = {"train_data": train_data, "val_data": val_data, "test_data": test_data}
    datamodule = hydra.utils.instantiate(config.datamodule, **params)
    model = hydra.utils.instantiate(config.model)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        log_every_n_steps=5,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    log.info("Testing")
    trainer.test(model=model, datamodule=datamodule)

    log.info("Done!")


if __name__ == "__main__":
    main()
