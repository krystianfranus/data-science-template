import numpy as np
import pytest

from src.model import LinearRegression
from src.preprocessing import split_data


def test_linear_regression_model(data):
    train_data, test_data = split_data(data, split_ratio=0.7)
    x_train, y_train = train_data["x"], train_data["y"]
    x_test, y_test = test_data["x"], test_data["y"]

    n_steps, lr = 100, 0.01
    model = LinearRegression(n_steps, lr)
    init_a, init_b = 0.0, 0.0
    model.a, model.b = init_a, init_b

    model.fit(x_train, y_train, x_test, y_test)
    params = model.get_params()
    fitted_a, fitted_b = params["a"], params["b"]

    a, b = 0.5, 1.0
    hparams = model.get_hparams()
    assert np.abs(fitted_a - a) < np.abs(init_a - a)
    assert np.abs(fitted_b - b) < np.abs(init_b - b)
    assert len(params) == 2
    assert len(hparams) == 2


@pytest.mark.parametrize(
    "n_steps,lr", [("100", 0.01), (100, "0.01"), (-100, 0.01), (100, -0.01)]
)
def test_incorrect_args_in_linear_regression_model(n_steps, lr):
    with pytest.raises((TypeError, ValueError)):
        LinearRegression(n_steps, lr)
