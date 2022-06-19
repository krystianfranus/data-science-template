import hydra
import mlflow as mf
import pandas as pd
from omegaconf import DictConfig

from src.model import LinearRegression


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(config: DictConfig):

    with mf.start_run(run_name="train"):
        # Load data
        train_data = pd.read_csv(config["train_path"])
        x_train = train_data["x"]
        y_train = train_data["y"]

        test_data = pd.read_csv(config["test_path"])
        x_test = test_data["x"]
        y_test = test_data["y"]

        # Train model
        n_steps = config["n_steps"]
        lr = config["lr"]
        model = LinearRegression(n_steps, lr)
        model.fit(x_train, y_train, x_test, y_test)

        # Log hyperparameters
        hparams = model.get_hparams()
        mf.log_params(hparams)

        # Log model (as artifact)
        mf.pyfunc.log_model("model", python_model=model)


if __name__ == "__main__":
    main()
