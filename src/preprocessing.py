from typing import Tuple

import pandas as pd


def split_data(
    data: pd.DataFrame,
    split_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test datasets.

    Parameters
    ----------
    data : DataFrame
        Dataset to be split.
    split_ratio : float
        Ratio of splitting. Must be between 0 and 1.

    Returns
    -------
    train_data : DataFrame
        Train dataset.
    test_data : DataFrame
        Test dataset.
    """
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between 0 and 1")

    train_data_size = int(len(data) * split_ratio)
    train_data = data[:train_data_size].reset_index(drop=True)
    test_data = data[train_data_size:].reset_index(drop=True)
    return train_data, test_data
