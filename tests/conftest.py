import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def data():
    n = 100
    x = np.random.uniform(-1, 1, size=n)
    err = np.random.normal(0, 0.2, n)

    a, b = 0.5, 1.0
    y = a * x + b + err

    data = pd.DataFrame({"x": x, "y": y})
    return data
