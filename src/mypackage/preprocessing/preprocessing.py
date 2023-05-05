import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


class ContentWise:
    def __init__(self, data_type: str) -> None:
        self.data_type = data_type
        self.train_data, self.val_data, self.test_data = None, None, None
        self.n_users, self.n_items = None, None

        # prefix = "s3://kf-north-bucket/data-science-template/data/contentwise/CW10M"
        prefix = Path("data/contentwise/data/contentwise/CW10M/")
        interactions_path = prefix / Path("interactions")
        impressions_dl_path = prefix / Path("impressions-direct-link")

        self.interactions = pd.concat(
            pd.read_parquet(p) for p in interactions_path.glob("*.parquet")
        ).reset_index()
        self.impressions_dl = pd.concat(
            pd.read_parquet(p) for p in impressions_dl_path.glob("*.parquet")
        ).reset_index()

    def prepare_data(self) -> None:
        match self.data_type:
            case "simple":
                self._prepare_simple()
            case "bpr":
                self._prepare_bpr()
            case _:
                raise ValueError(f"Invalid data type, you provided '{self.data_type}'")

    def _prepare_simple(self) -> None:
        interactions = self._common()

        # Split data
        self.train_data = interactions[
            interactions["timestamp"] < dt.datetime(2019, 4, 14)
        ]
        self.val_data = interactions[
            interactions["timestamp"] >= dt.datetime(2019, 4, 14)
        ]

        # Prepare user/item to idx mappers based on train data
        unique_users = self.train_data["user"].unique()
        unique_items = self.train_data["item"].unique()
        self.n_users = unique_users.size
        self.n_items = unique_items.size
        train_user_to_idx = pd.DataFrame(
            {"user": unique_users, "user_idx": np.arange(self.n_users)}
        )
        train_item_to_idx = pd.DataFrame(
            {"item": unique_items, "item_idx": np.arange(self.n_items)}
        )

        # Map user/item to idx
        self.train_data = self.train_data.merge(
            train_user_to_idx, on="user", how="inner"
        )
        self.train_data = self.train_data.merge(
            train_item_to_idx, on="item", how="inner"
        )
        self.val_data = self.val_data.merge(train_user_to_idx, on="user", how="inner")
        self.val_data = self.val_data.merge(train_item_to_idx, on="item", how="inner")

        self.train_data = self.train_data.sort_values("timestamp").reset_index(
            drop=True
        )
        self.val_data = self.val_data.sort_values("timestamp").reset_index(drop=True)

        # Select valid columns
        self.train_data = self.train_data[["user_idx", "item_idx", "target"]]
        self.train_data.columns = ["user", "item", "target"]
        self.val_data = self.val_data[["user_idx", "item_idx", "target"]]
        self.val_data.columns = ["user", "item", "target"]

        # Mock test_data
        self.test_data = (
            self.val_data.copy()
        )  # test set == validation set (to change in the future!)

    def _prepare_bpr(self) -> None:
        interactions = self._common()

        # Split data
        self.train_data = interactions[
            interactions["timestamp"] < dt.datetime(2019, 4, 14)
        ]
        tmp0 = self.train_data.loc[self.train_data["target"] == 0, ["user", "item"]]
        tmp1 = self.train_data.loc[self.train_data["target"] == 1, ["user", "item"]]
        self.train_data = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
        # self.train_data = self.train_data.sample(frac=0.2, random_state=0).reset_index(drop=True)  # noqa
        self.val_data = interactions[
            interactions["timestamp"] >= dt.datetime(2019, 4, 14)
        ]

        # Prepare user/item to idx mappers based on train data
        unique_users = self.train_data["user"].unique()
        item_neg_set = set(self.train_data["item_neg"])
        item_pos_set = set(self.train_data["item_pos"])
        unique_items = pd.Series(list(item_neg_set | item_pos_set)).unique()
        self.n_users = unique_users.size
        self.n_items = unique_items.size
        train_user_to_idx = pd.DataFrame(
            {"user": unique_users, "user_idx": np.arange(self.n_users)}
        )
        train_item_to_idx = pd.DataFrame(
            {"item": unique_items, "item_idx": np.arange(self.n_items)}
        )

        # Map user/item to idx and handle column names conflicts
        self.train_data = self.train_data.merge(
            train_user_to_idx, on="user", how="inner"
        )
        self.train_data = self.train_data[["user_idx", "item_neg", "item_pos"]].rename(
            columns={"user_idx": "user"}
        )
        self.train_data = self.train_data.merge(
            train_item_to_idx, left_on="item_neg", right_on="item", how="inner"
        )
        self.train_data = self.train_data[["user", "item_idx", "item_pos"]].rename(
            columns={"item_idx": "item_neg"}
        )
        self.train_data = self.train_data.merge(
            train_item_to_idx, left_on="item_pos", right_on="item", how="inner"
        )
        self.train_data = self.train_data[["user", "item_neg", "item_idx"]].rename(
            columns={"item_idx": "item_pos"}
        )

        self.val_data = self.val_data.merge(train_user_to_idx, on="user", how="inner")
        self.val_data = self.val_data.merge(train_item_to_idx, on="item", how="inner")
        self.val_data = self.val_data[["user_idx", "item_idx", "target"]].rename(
            columns={"user_idx": "user", "item_idx": "item"}
        )

        # Mock test_data
        self.test_data = (
            self.val_data.copy()
        )  # test set == validation set (to change in the future!)

    def _common(self) -> pd.DataFrame:
        # Select movies only from other item types
        interactions = self.interactions[self.interactions["item_type"] == 0]
        # Select clicks only from other interaction types
        interactions = interactions[interactions["interaction_type"] == 0]

        interactions["utc_ts_milliseconds"] = pd.to_datetime(
            interactions["utc_ts_milliseconds"], unit="ms"
        )

        impressions_dl = self.impressions_dl.explode("recommended_series_list")
        impressions_dl["recommended_series_list"] = pd.to_numeric(
            impressions_dl["recommended_series_list"]
        )

        # Join positive (clicks) interactions with negative (impressions)
        interactions = interactions.merge(impressions_dl, "inner", "recommendation_id")

        # Mark positive interactions with 1 and negative with 0
        interactions.loc[
            interactions["series_id"] == interactions["recommended_series_list"],
            "target",
        ] = 1
        interactions.loc[
            interactions["series_id"] != interactions["recommended_series_list"],
            "target",
        ] = 0
        interactions["target"] = interactions["target"].astype("int32")

        interactions = interactions[
            ["utc_ts_milliseconds", "user_id", "recommended_series_list", "target"]
        ]
        interactions.columns = ["timestamp", "user", "item", "target"]

        # Handle (user, item) duplicates
        interactions = (
            interactions.groupby(["user", "item"])
            .agg({"target": "sum", "timestamp": "max"})
            .reset_index()
        )
        interactions.loc[interactions["target"] > 0, "target"] = 1

        # Get rid of inactive users and items
        tmp_u = (
            interactions.groupby("user")
            .agg({"target": "sum"})
            .rename(columns={"target": "sum"})
            .reset_index()
        )
        tmp_u = tmp_u[tmp_u["sum"] >= 5]
        interactions = interactions.merge(tmp_u, "inner", "user")
        tmp_i = (
            interactions.groupby("item")
            .agg({"target": "sum"})
            .rename(columns={"target": "sum"})
            .reset_index()
        )
        tmp_i = tmp_i[tmp_i["sum"] >= 10]
        interactions = interactions.merge(tmp_i, "inner", "item")
        interactions = interactions[["user", "item", "target", "timestamp"]]

        return interactions
