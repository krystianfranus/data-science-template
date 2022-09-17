import logging

import hydra
import pandas as pd
from clearml import Task
from omegaconf import DictConfig

from src.preprocessing.preprocessing import split_data

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/preprocessing/", config_name="config"
)
def main(config: DictConfig):
    task = Task.init(project_name="ds_template", task_name="preprocessing")

    # Load raw data
    raw_data_path = config.raw_data_path
    data = pd.read_csv(raw_data_path)

    # Split data to train/test
    log.info("Data splitting")
    train_data, val_data, test_data = split_data(data, config.split_ratio)

    # Upload artifacts
    task.upload_artifact("train_data", train_data)
    task.upload_artifact("val_data", val_data)
    task.upload_artifact("test_data", test_data)

    log.info("Done!")


if __name__ == "__main__":
    main()
