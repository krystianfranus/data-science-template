import logging

import hydra
import mlflow as mf
import pandas as pd
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.model import LinearRegression

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="train")
def main(config: DictConfig):
    with mf.start_run(run_name="train"):
        hconfig = HydraConfig.get()

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
        log.info(f"Training")
        n_steps = config.n_steps
        lr = config.lr
        model = LinearRegression(n_steps, lr)
        model.fit(x_train, y_train, x_test, y_test)

        # Log hyperparameters
        log.info("Logging (mlflow) parameters")
        hparams = model.get_hparams()
        mf.log_params(hparams)

        # Log artifacts
        log.info("Logging model (as artifact)")
        mf.pyfunc.log_model("model", python_model=model)

        mf.log_artifacts(hconfig.runtime.output_dir, artifact_path="logs/train")


if __name__ == "__main__":
    main()
