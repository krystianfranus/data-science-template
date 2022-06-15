import pandas as pd
from mlflow import log_params

from src.model import LinearRegression


def main():
    # Load data
    train_data = pd.read_csv("data/train.csv")
    x_train = train_data["x_train"]
    y_train = train_data["y_train"]

    test_data = pd.read_csv("data/test.csv")
    x_test = test_data["x_test"]
    y_test = test_data["y_test"]

    # Train model
    model = LinearRegression(n_steps=200, lr=0.01)
    model.fit(x_train, y_train, x_test, y_test)

    # Log hyperparameters
    hparams = model.get_hparams()
    log_params(hparams)


if __name__ == "__main__":
    main()
