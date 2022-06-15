import numpy as np


def mse(y_true, y_pred):
    err = y_true - y_pred
    value = np.mean(err**2)
    return value
