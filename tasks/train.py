import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.model import LinearRegression

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="train")
def main(config: DictConfig):

    # Load train and test data
    train_data_path = config.train_data_path
    log.info(f"Loading train data from '{train_data_path}'")
    train_data = pd.read_csv(train_data_path)
    x_train = train_data["x"]
    y_train = train_data["y"]

    test_data_path = config.test_data_path
    log.info(f"Loading test data from '{test_data_path}'")
    test_data = pd.read_csv(test_data_path)
    x_test = test_data["x"]
    y_test = test_data["y"]

    # Train model
    log.info("Training")
    n_steps = config.n_steps
    lr = config.lr
    model = LinearRegression(n_steps, lr)
    model.fit(x_train, y_train)


if __name__ == "__main__":
    main()
