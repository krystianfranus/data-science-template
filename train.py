import hydra
import mlflow as mf
import pandas as pd
from omegaconf import DictConfig

from src.model import LinearRegression


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(config: DictConfig):

    # mf.set_experiment("my model")
    with mf.start_run() as mlrun:
        # Load data
        train_data = pd.read_csv("data/train.csv")
        x_train = train_data["x_train"]
        y_train = train_data["y_train"]

        test_data = pd.read_csv("data/test.csv")
        x_test = test_data["x_test"]
        y_test = test_data["y_test"]

        # Train model
        n_steps = config["n_steps"]
        lr = config["lr"]
        model = LinearRegression(n_steps, lr)
        model.fit(x_train, y_train, x_test, y_test)

        # Log hyperparameters
        hparams = model.get_hparams()
        mf.log_params(hparams)

        # Save model
        mf.pyfunc.log_model("model", python_model=model)


if __name__ == "__main__":
    main()
