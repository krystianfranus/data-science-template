from typing import Tuple

import pandas as pd


def prepare_explicit_data(
    ratings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = ratings[:700_000].reset_index(drop=True)
    val_data = ratings[700_000:850_000].reset_index(drop=True)
    test_data = ratings[850_000:].reset_index(drop=True)
    data = pd.concat((train_data, val_data, test_data)).reset_index(drop=True)

    user_to_idx = {user: idx for idx, user in enumerate(data["user"].unique())}
    item_to_idx = {item: idx for idx, item in enumerate(data["item"].unique())}

    train_data["user"] = train_data["user"].map(user_to_idx)
    val_data["user"] = val_data["user"].map(user_to_idx)
    test_data["user"] = test_data["user"].map(user_to_idx)
    train_data["item"] = train_data["item"].map(item_to_idx)
    val_data["item"] = val_data["item"].map(item_to_idx)
    test_data["item"] = test_data["item"].map(item_to_idx)

    return train_data, val_data, test_data


def prepare_implicit_data(
    ratings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings.loc[ratings["rating"] < 4, "rating"] = 0
    ratings.loc[ratings["rating"] >= 4, "rating"] = 1
    train_data = ratings[:700_000].reset_index(drop=True)
    val_data = ratings[700_000:850_000].reset_index(drop=True)
    test_data = ratings[850_000:].reset_index(drop=True)
    data = pd.concat((train_data, val_data, test_data)).reset_index(drop=True)

    user_to_idx = {user: idx for idx, user in enumerate(data["user"].unique())}
    item_to_idx = {item: idx for idx, item in enumerate(data["item"].unique())}

    train_data["user"] = train_data["user"].map(user_to_idx)
    val_data["user"] = val_data["user"].map(user_to_idx)
    test_data["user"] = test_data["user"].map(user_to_idx)
    train_data["item"] = train_data["item"].map(item_to_idx)
    val_data["item"] = val_data["item"].map(item_to_idx)
    test_data["item"] = test_data["item"].map(item_to_idx)

    return train_data, val_data, test_data
