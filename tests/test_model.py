import numpy as np

from src.model import LinearRegression
from src.preprocessing import split_data


def test_linear_regression_model(data):
    train_data, test_data = split_data(data, split_ratio=0.7)
    x_train, y_train = train_data["x"], train_data["y"]
    x_test, y_test = test_data["x"], test_data["y"]

    model = LinearRegression(n_steps=100, lr=0.01)
    params = model.get_params()
    init_a, init_b = params["a"], params["b"]

    model.fit(x_train, y_train, x_test, y_test)
    params = model.get_params()
    fitted_a, fitted_b = params["a"], params["b"]

    a, b = 0.5, 1.0
    assert np.abs(fitted_a - a) < np.abs(init_a - a)
    assert np.abs(fitted_b - b) < np.abs(init_b - b)
