import logging
from pathlib import Path

import hydra
import mlflow as mf
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.preprocessing import split_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="preprocess")
def main(config: DictConfig):
    with mf.start_run(run_name="preprocess") as active_run:
        hconfig = HydraConfig.get()

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

        # Log params
        log.info("Logging (mlflow) parameters")
        mf.log_param("split_ratio", config.split_ratio)

        # Log artifacts
        log.info(f"Logging (mlflow) artifacts into '{active_run.info.artifact_uri}'")
        mf.log_artifacts(config.processed_data_dir, artifact_path="data")
        mf.log_artifacts(hconfig.runtime.output_dir, artifact_path="logs/preprocess")


if __name__ == "__main__":
    main()
