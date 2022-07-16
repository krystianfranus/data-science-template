import numpy as np
from mlflow import log_metric
from mlflow.pyfunc import PythonModel
from numpy import ndarray

from src.metrics import mse


class LinearRegression(PythonModel):
    """Simple linear regression model for one dimensional tasks.

    Parameters
    ----------
    n_steps : int
        N steps in learning process.
    lr : float
        Learning rate.

    Examples
    --------
    >>> # Prepare data
    >>> x_train = np.array([1., 2., 3.])
    >>> y_train = np.array([2., 3., 4.])
    >>> x_test = np.array([4.])
    >>> y_test = np.array([5.])
    >>> # Fit the model
    >>> model = LinearRegression(n_steps=200, lr=0.01)
    >>> model.fit(x_train, y_train, x_test, y_test)
    """

    def __init__(self, n_steps: int, lr: float):
        self.n_steps = n_steps
        self.lr = lr
        self.a = np.random.uniform()
        self.b = np.random.uniform()

    def fit(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray):
        """Fit parameters.

        Parameters
        ----------
        x_train : ndarray
            Train data.
        y_train : ndarray
            Train labels (ground-truths).
        x_test : ndarray
            Test data.
        y_test : ndarray
            Test labels (ground-truths).
        """
        n = len(x_train)

        for step in range(self.n_steps):
            # Predict
            y_train_pred = self.score(x_train)
            y_test_pred = self.score(x_test)

            # Log metrics
            log_metric("train/mse", mse(y_train, y_train_pred), step=step)
            log_metric("test/mse", mse(y_test, y_test_pred), step=step)

            # Optimize (in terms of mse) params with gradient descent
            self.a -= self.lr * -2 / n * np.sum(x_train * (y_train - y_train_pred))
            self.b -= self.lr * -2 / n * np.sum(y_train - y_train_pred)

    def predict(self, context, model_input):
        x_test = model_input.to_numpy()
        return self.score(x_test)

    def score(self, x_test: ndarray):
        """Evaluate model for given input test data.

        Parameters
        ----------
        x_test : ndarray
            Test data.
        """
        return self.a * x_test + self.b

    def get_params(self):
        """Get model parameters."""
        return {"a": self.a, "b": self.b}

    def get_hparams(self):
        """Get model hyperparameters."""
        return {"n_steps": self.n_steps, "lr": self.lr}
