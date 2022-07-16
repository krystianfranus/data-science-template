import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def data():
    x = np.random.normal(size=20)
    y = np.random.normal(size=20)
    data = pd.DataFrame({"x": x, "y": y})
    return data
