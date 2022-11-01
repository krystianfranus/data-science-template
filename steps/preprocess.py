import logging

import hydra
import pandas as pd
from clearml import Task, TaskTypes
from omegaconf import DictConfig

from src.preprocessing.preprocessing import prepare_explicit_data, prepare_implicit_data

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/preprocessing/", config_name="config"
)
def main(config: DictConfig):
    task = Task.init(
        project_name="My project",
        task_name="Preprocessing",
        task_type=TaskTypes.data_processing,
        output_uri="s3://kfranus-bucket/data",
    )

    # if config.execute_remotely:
    #     task.execute_remotely()

    log.info("[My Logger] Data loading")
    raw_data_path = config.raw_data_path
    ratings = pd.read_csv(
        raw_data_path,
        sep="::",
        names=["user", "item", "rating", "timestamp"],
        engine="python",
    )
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    log.info("[My Logger] Data splitting")
    train_data, val_data, test_data = None, None, None
    if config.data_type == "explicit":
        train_data, val_data, test_data = prepare_explicit_data(ratings)
    elif config.data_type == "implicit":
        train_data, val_data, test_data = prepare_implicit_data(ratings)

    log.info("[My Logger] Data (artifacts) uploading")
    task.upload_artifact("train_data", train_data)
    task.upload_artifact("val_data", val_data)
    task.upload_artifact("test_data", test_data)

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
