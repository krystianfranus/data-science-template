import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from clearml import Task
from src.preprocessing import split_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="preprocess")
def main(config: DictConfig):
    task = Task.init(project_name="ds_template", task_name="preprocessing")

    # Load raw data
    raw_data_path = config.raw_data_path
    log.info(f"Loading raw data from '{raw_data_path}'")
    data = pd.read_csv(raw_data_path)

    # Split data to train/test
    log.info("Splitting data")
    train_data, test_data = split_data(data, config.split_ratio)

    # Save train and test data
    Path(config.processed_data_dir).mkdir(parents=True, exist_ok=True)

    train_data_path = config.train_data_path
    log.info(f"Saving train data into '{train_data_path}'")
    train_data.to_csv(train_data_path, index=False)

    test_data_path = config.test_data_path
    log.info(f"Saving test data into '{test_data_path}'")
    test_data.to_csv(test_data_path, index=False)


if __name__ == "__main__":
    main()
