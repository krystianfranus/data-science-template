import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl


class ContentWise:
    def __init__(self, data_type):
        # prefix = "s3://kf-north-bucket/data-science-template/data/contentwise/CW10M"
        prefix = Path("data/contentwise/data/contentwise/CW10M/")

        self.data_type = data_type
        interactions_path = prefix / Path("interactions")
        impressions_dl_path = prefix / Path("impressions-direct-link")

        self.interactions = pd.concat(
            pd.read_parquet(p) for p in interactions_path.glob("*.parquet")
        ).reset_index()
        self.impressions_dl = pd.concat(
            pd.read_parquet(p) for p in impressions_dl_path.glob("*.parquet")
        ).reset_index()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "implicit":
            train_data, val_data, test_data, n_users, n_items = self._prepare_implicit()
        elif self.data_type == "implicit_bpr":
            train_data, val_data, test_data, n_users, n_items = self._prepare_implicit_bpr()
        return train_data, val_data, test_data, n_users, n_items

    # def _prepare_implicit_pl(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     # Select 'clicks' only from all interactions
    #     interactions = self.interactions.filter(pl.col("interaction_type") == 0)
    #
    #     impressions_dl = self.impressions_dl.explode("recommended_series_list")
    #
    #     # Join indirectly positive actions with negative (impressions)
    #     interactions = interactions.join(
    #         impressions_dl, on="recommendation_id", how="inner"
    #     )
    #
    #     # Mark positive interactions with 1 and negative with 0
    #     interactions = interactions.with_column(
    #         pl.when(pl.col("series_id") == pl.col("recommended_series_list"))
    #         .then(1)
    #         .otherwise(0)
    #         .alias("target")
    #     )
    #     interactions = interactions.rename(
    #         {
    #             "user_id": "user",
    #             "recommended_series_list": "item",
    #             "utc_ts_milliseconds": "timestamp",
    #         }
    #     )
    #     interactions = interactions.with_column(
    #         pl.col("timestamp").cast(pl.Datetime).dt.with_time_unit("ms")
    #     )
    #     # Handle (user, item) duplicates
    #     interactions = interactions.groupby(["user", "item"]).agg(
    #         [pl.sum("target"), pl.max("timestamp")]
    #     )
    #     interactions = interactions.with_column(
    #         pl.when(pl.col("target") > 0).then(1).otherwise(0).alias("target")
    #     )
    #     interactions = interactions.sort("timestamp")
    #     interactions = interactions.cache()
    #
    #     # Split data
    #     train_data = interactions.filter(pl.col("timestamp") < dt.date(2019, 4, 14))
    #     val_data = interactions.filter(pl.col("timestamp") >= dt.date(2019, 4, 14))
    #
    #     # Prepare user/item to idx mappers based on train data
    #     train_user_to_idx = train_data.select(
    #         [
    #             pl.col("user").unique(),
    #             pl.col("user").unique().rank().cast(pl.Int64).alias("user_idx") - 1,
    #         ]
    #     )
    #     train_item_to_idx = train_data.select(
    #         [
    #             pl.col("item").unique(),
    #             pl.col("item").unique().rank().cast(pl.Int64).alias("item_idx") - 1,
    #         ]
    #     )
    #
    #     # Map user/item to idx
    #     train_data = train_data.join(train_user_to_idx, on="user", how="inner")
    #     train_data = train_data.join(train_item_to_idx, on="item", how="inner")
    #     val_data = val_data.join(train_user_to_idx, on="user", how="inner")
    #     val_data = val_data.join(train_item_to_idx, on="item", how="inner")
    #
    #     # Select valid columns
    #     train_data = train_data.select(
    #         [
    #             pl.col("user_idx").alias("user"),
    #             pl.col("item_idx").alias("item"),
    #             "target",
    #         ]
    #     )
    #     val_data = val_data.select(
    #         [
    #             pl.col("user_idx").alias("user"),
    #             pl.col("item_idx").alias("item"),
    #             "target",
    #         ]
    #     )
    #     test_data = val_data  # test set == validation set (to change in the future!)
    #
    #     return train_data.collect(), val_data.collect(), test_data.collect()

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
        # Select 'clicks' only from all interactions
        interactions = self.interactions[
            self.interactions["interaction_type"] == 0
        ].reset_index(drop=True)

        impressions_dl = self.impressions_dl.explode("recommended_series_list")
        impressions_dl["recommended_series_list"] = pd.to_numeric(
            impressions_dl["recommended_series_list"]
        )

        # Join indirectly positive actions with negative (impressions)
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

        interactions = interactions[
            ["user_id", "recommended_series_list", "target", "utc_ts_milliseconds"]
        ]
        interactions.columns = ["user", "item", "target", "timestamp"]
        interactions["target"] = interactions["target"].astype("int32")
        interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="ms")

        # Handle (user, item) duplicates
        interactions = (
            interactions.groupby(["user", "item"])
            .agg({"target": "sum", "timestamp": "max"})
            .reset_index()
        )
        interactions.loc[interactions["target"] > 0, "target"] = 1

        interactions = interactions.sort_values("timestamp").reset_index(drop=True)

        # Split data
        train_data = interactions[
            interactions["timestamp"] < dt.datetime(2019, 4, 14)
        ].reset_index(drop=True)
        val_data = interactions[
            interactions["timestamp"] >= dt.datetime(2019, 4, 14)
        ].reset_index(drop=True)

        # Prepare user/item to idx mappers based on train data
        unique_users = np.sort(train_data["user"].unique())
        unique_items = np.sort(train_data["item"].unique())
        train_user_to_idx = pd.DataFrame(
            {"user": unique_users, "user_idx": np.arange(unique_users.size)}
        )
        train_item_to_idx = pd.DataFrame(
            {"item": unique_items, "item_idx": np.arange(unique_items.size)}
        )

        # Map user/item to idx
        train_data = train_data.merge(train_user_to_idx, on="user", how="inner")
        train_data = train_data.merge(train_item_to_idx, on="item", how="inner")
        val_data = val_data.merge(train_user_to_idx, on="user", how="inner")
        val_data = val_data.merge(train_item_to_idx, on="item", how="inner")

        train_data = train_data.sort_values("timestamp").reset_index(drop=True)
        val_data = val_data.sort_values("timestamp").reset_index(drop=True)

        # Select valid columns
        train_data = train_data[["user_idx", "item_idx", "target"]]
        train_data.columns = ["user", "item", "target"]
        val_data = val_data[["user_idx", "item_idx", "target"]]
        val_data.columns = ["user", "item", "target"]

        test_data = (
            val_data.copy()
        )  # test set == validation set (to change in the future!)

        n_users = unique_users.size
        n_items = unique_items.size
        return train_data, val_data, test_data, n_users, n_items

    def _prepare_implicit_bpr(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
        interactions = self.interactions[
            self.interactions["interaction_type"] == 0
        ].reset_index(drop=True)

        impressions_dl = self.impressions_dl.explode("recommended_series_list")
        impressions_dl["recommended_series_list"] = pd.to_numeric(
            impressions_dl["recommended_series_list"]
        )

        # Join indirectly positive actions with negative (impressions)
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

        interactions = interactions[
            ["user_id", "recommended_series_list", "target", "utc_ts_milliseconds"]
        ]
        interactions.columns = ["user", "item", "target", "timestamp"]
        interactions["target"] = interactions["target"].astype("int32")
        interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="ms")

        # Handle (user, item) duplicates
        interactions = (
            interactions.groupby(["user", "item"])
            .agg({"target": "sum", "timestamp": "max"})
            .reset_index()
        )
        interactions.loc[interactions["target"] > 0, "target"] = 1

        interactions = interactions.sort_values("timestamp").reset_index(drop=True)

        # Split data into train/val/test
        train_data = interactions[:1_000_000].reset_index(drop=True)
        tmp0 = train_data.loc[train_data["target"] == 0, ["user", "item"]]
        tmp1 = train_data.loc[train_data["target"] == 1, ["user", "item"]]
        train_data = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
        # train_data = train_data.sample(frac=0.2, random_state=0).reset_index(drop=True)  # noqa
        val_data = interactions[1_000_000:1_100_000].reset_index(drop=True)
        test_data = interactions[1_100_000:].reset_index(drop=True)

        # Prepare unique train user and items
        train_users = train_data["user"].unique()
        item_neg_set = set(train_data["item_neg"])
        item_pos_set = set(train_data["item_pos"])
        train_items = pd.Series(list(item_neg_set | item_pos_set)).unique()

        # Filter val/test data
        val_data = val_data[val_data["user"].isin(train_users)]
        val_data = val_data[val_data["item"].isin(train_items)]
        val_data = val_data.reset_index(drop=True)
        test_data = test_data[test_data["user"].isin(train_users)]
        test_data = test_data[test_data["item"].isin(train_items)]
        test_data = test_data.reset_index(drop=True)

        # Map idx
        user_to_idx = {user: idx for idx, user in enumerate(train_users)}
        item_to_idx = {item: idx for idx, item in enumerate(train_items)}
        train_data["user"] = train_data["user"].map(user_to_idx)
        train_data["item_neg"] = train_data["item_neg"].map(item_to_idx)
        train_data["item_pos"] = train_data["item_pos"].map(item_to_idx)
        val_data["user"] = val_data["user"].map(user_to_idx)
        val_data["item"] = val_data["item"].map(item_to_idx)
        test_data["user"] = test_data["user"].map(user_to_idx)
        test_data["item"] = test_data["item"].map(item_to_idx)

        n_users = train_users.size
        n_items = train_items.size
        return train_data, val_data, test_data, n_users, n_items

    def save_data(self, train_data, val_data, test_data):
        train_data.to_parquet(
            f"data/contentwise/train_data_{self.data_type}.parquet", index=False
        )
        val_data.to_parquet(
            f"data/contentwise/val_data_{self.data_type}.parquet", index=False
        )
        test_data.to_parquet(
            f"data/contentwise/test_data_{self.data_type}.parquet", index=False
        )
