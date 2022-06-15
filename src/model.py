import numpy as np
from mlflow import log_metric

from src.metrics import mse


class LinearRegression:
    def __init__(self, n_steps, lr):
        self.n_steps = n_steps
        self.lr = lr
        self.a = np.random.uniform()
        self.b = np.random.uniform()

    def fit(self, x_train, y_train, x_test, y_test):
        n = len(x_train)

        for step in range(self.n_steps):
            # Predict
            y_train_pred = self.predict(x_train)
            y_test_pred = self.predict(x_test)

            # Log metrics
            log_metric("train/mse", mse(y_train, y_train_pred), step=step)
            log_metric("test/mse", mse(y_test, y_test_pred), step=step)

            # Optimize (in terms of mse) params with gradient descent
            self.a -= self.lr * -2 / n * np.sum(x_train * (y_train - y_train_pred))
            self.b -= self.lr * -2 / n * np.sum(y_train - y_train_pred)

    def predict(self, x_test):
        return self.a * x_test + self.b

    def get_params(self):
        return {"a": self.a, "b": self.b}

    def get_hparams(self):
        return {"n_steps": self.n_steps, "lr": self.lr}
