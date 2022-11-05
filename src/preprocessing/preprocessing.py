from typing import Tuple

import pandas as pd


class MovieLens1M:
    def __init__(self, remote_data, data_type):
        if remote_data:
            prefix = "s3://kfranus-bucket"
            data_path = f"{prefix}/data-science-template/data/movielens/ratings.dat"
        else:
            data_path = "data/movielens/ratings.dat"

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

    def save_data(self, train_data, val_data, test_data):
        train_data.to_csv(
            f"data/movielens/train_data_{self.data_type}.csv", index=False
        )
        val_data.to_csv(f"data/movielens/val_data_{self.data_type}.csv", index=False)
        test_data.to_csv(f"data/movielens/test_data_{self.data_type}.csv", index=False)


class ContentWise:
    def __init__(self, remote_data, data_type):
        if remote_data:  # TODO
            prefix = "s3://kfranus-bucket"
            data_path = f"{prefix}/data-science-template/data/movielens/ratings.dat"
            data_path2 = f"{prefix}/data-science-template/data/movielens/ratings.dat"
        else:
            prefix = "data/contentwise/data/contentwise"
            data_path = f"{prefix}/CW10M-CSV/interactions.csv.gz"
            data_path2 = f"{prefix}/CW10M-CSV/impressions-direct-link.csv.gz"

        self.data = pd.read_csv(data_path)
        self.data2 = pd.read_csv(data_path2)
        self.data_type = data_type

    def preprocess(self):
        pass

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "implicit":
            train_data, val_data, test_data = self._prepare_implicit()
        return train_data, val_data, test_data

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = self.data[self.data["interaction_type"] == 0].reset_index(drop=True)
        columns = [
            "utc_ts_milliseconds",
            "user_id",
            "series_id",
            "recommendation_id",
            "vision_factor",
        ]
        self.data = self.data[columns]

        self.data2["recommended_series_list"] = (
            self.data2["recommended_series_list"]
            .str.replace(r"(\[|\])", "", regex=True)
            .str.split()
        )
        self.data2 = self.data2.explode("recommended_series_list").reset_index(
            drop=True
        )

        merged = self.data.merge(self.data2, "inner", "recommendation_id")
        merged["recommended_series_list"] = pd.to_numeric(
            merged["recommended_series_list"]
        )
        merged.loc[
            merged["series_id"] == merged["recommended_series_list"], "target"
        ] = 1
        merged.loc[
            merged["series_id"] != merged["recommended_series_list"], "target"
        ] = 0
        merged = merged[["user_id", "series_id", "target", "utc_ts_milliseconds"]]
        merged["target"] = merged["target"].astype(int)
        merged.columns = ["user", "item", "target", "timestamp"]

        user_to_idx = {user: idx for idx, user in enumerate(merged["user"].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(merged["item"].unique())}
        merged["user"] = merged["user"].map(user_to_idx)
        merged["item"] = merged["item"].map(item_to_idx)

        train_data = merged[:2_500_000].reset_index(drop=True)
        val_data = merged[2_500_000:3_000_000].reset_index(drop=True)
        test_data = merged[3_000_000:].reset_index(drop=True)

        return train_data, val_data, test_data

    def save_data(self, train_data, val_data, test_data):
        train_data.to_csv(
            f"data/contentwise/train_data_{self.data_type}.csv", index=False
        )
        val_data.to_csv(f"data/contentwise/val_data_{self.data_type}.csv", index=False)
        test_data.to_csv(
            f"data/contentwise/test_data_{self.data_type}.csv", index=False
        )
