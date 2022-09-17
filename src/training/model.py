import numpy as np
from numpy import ndarray


class LinearRegression:
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

    @property
    def n_steps(self):
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value):
        if not isinstance(value, int):
            raise TypeError("n_steps must be of an integer type")
        if value <= 0:
            raise ValueError("n_steps must be positive")
        self._n_steps = value

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if not isinstance(value, float):
            raise TypeError("lr must be of a float type")
        if value <= 0:
            raise ValueError("lr must be positive")
        self._lr = value

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
    ):
        """Fit parameters.

        Parameters
        ----------
        x_train : ndarray
            Train data.
        y_train : ndarray
            Train labels (ground-truths).
        """
        n = len(x_train)

        for step in range(self.n_steps):
            # Predict
            y_train_pred = self.score(x_train)

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
