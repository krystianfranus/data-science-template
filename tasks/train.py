import logging

import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig

from src.training.datamodules.datamodule import SimpleDataModule
from src.training.models.model import LinearRegression

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="train")
def main(config: DictConfig):
    Task.init(project_name="ds_template", task_name="training")
    task2 = Task.get_task(task_id="210b86a45a71478f8c40c12ae94ea76c")

    # Load train data
    train_data = task2.artifacts["train_data"].get()

    # Load val data
    val_data = task2.artifacts["test_data"].get()

    # Load test data
    test_data = val_data.copy()

    # MAIN CODE
    datamodule = SimpleDataModule(train_data, val_data, test_data)
    model = LinearRegression(lr=config.lr)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        log_every_n_steps=5,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
