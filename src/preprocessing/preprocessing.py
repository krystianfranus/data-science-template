from typing import Tuple

import pandas as pd


class MovieLens1M:
    def __init__(self, data_path, data_type):
        self.data = pd.read_csv(
            data_path,
            sep="::",
            names=["user", "item", "target", "timestamp"],
            engine="python",
        )
        self.data_type = data_type

    def preprocess(self):
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "explicit":
            train_data, val_data, test_data = self._prepare_explicit()
        elif self.data_type == "implicit":
            train_data, val_data, test_data = self._prepare_implicit()
        elif self.data_type == "implicit_bpr":
            train_data, val_data, test_data = self._prepare_implicit_bpr()
        return train_data, val_data, test_data

    def _prepare_explicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        user_to_idx = {user: idx for idx, user in enumerate(self.data["user"].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(self.data["item"].unique())}
        self.data["user"] = self.data["user"].map(user_to_idx)
        self.data["item"] = self.data["item"].map(item_to_idx)

        train_data = self.data[:700_000].reset_index(drop=True)
        val_data = self.data[700_000:850_000].reset_index(drop=True)
        test_data = self.data[850_000:].reset_index(drop=True)

        return train_data, val_data, test_data

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data.loc[self.data["target"] < 4, "target"] = 0
        self.data.loc[self.data["target"] >= 4, "target"] = 1

        user_to_idx = {user: idx for idx, user in enumerate(self.data["user"].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(self.data["item"].unique())}
        self.data["user"] = self.data["user"].map(user_to_idx)
        self.data["item"] = self.data["item"].map(item_to_idx)

        train_data = self.data[:700_000].reset_index(drop=True)
        val_data = self.data[700_000:850_000].reset_index(drop=True)
        test_data = self.data[850_000:].reset_index(drop=True)

        return train_data, val_data, test_data

    def _prepare_implicit_bpr(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        user_to_idx = {user: idx for idx, user in enumerate(self.data["user"].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(self.data["item"].unique())}
        self.data["user"] = self.data["user"].map(user_to_idx)
        self.data["item"] = self.data["item"].map(item_to_idx)

        self.data.loc[self.data["target"] < 4, "target"] = 0
        self.data.loc[self.data["target"] >= 4, "target"] = 1

        train_data = self.data[:700_000].reset_index(drop=True)
        tmp0 = train_data.loc[train_data["target"] == 0, ["user", "item"]]
        tmp1 = train_data.loc[train_data["target"] == 1, ["user", "item"]]
        train_data = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
        train_data = train_data.sample(frac=0.05, random_state=0).reset_index(drop=True)
        val_data = self.data[700_000:850_000].reset_index(drop=True)
        test_data = self.data[850_000:].reset_index(drop=True)

        return train_data, val_data, test_data
