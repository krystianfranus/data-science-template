from typing import Tuple

import pandas as pd


def split_data(
    data: pd.DataFrame,
    split_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data_size = int(len(data) * split_ratio)
    rest_data_size = len(data) - train_data_size

    train_data = data[:train_data_size]
    train_data = train_data.reset_index(drop=True)
    val_data = data[train_data_size : train_data_size + rest_data_size // 2]
    val_data = val_data.reset_index(drop=True)
    test_data = data[train_data_size + rest_data_size // 2 :]
    test_data = test_data.reset_index(drop=True)

    return train_data, val_data, test_data
