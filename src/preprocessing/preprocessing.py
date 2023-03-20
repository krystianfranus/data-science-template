import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl


class ContentWise:
    def __init__(self, data_type, use_polars):
        # # Reading data from s3 with polars is inappropriate!
        # prefix = "s3://kf-north-bucket/data-science-template/data/contentwise/CW10M"
        prefix = "data/contentwise/data/contentwise/CW10M"

        self.data_type = data_type
        self.use_polars = use_polars
        if use_polars:
            interactions_path = f"{prefix}/interactions/*.parquet"
            impressions_dl_path = f"{prefix}/impressions-direct-link/*.parquet"
            self.interactions = pl.scan_parquet(interactions_path)
            self.impressions_dl = pl.scan_parquet(impressions_dl_path)
        else:
            interactions_path = Path(prefix) / Path("interactions")
            self.interactions = pd.concat(
                pd.read_parquet(p) for p in interactions_path.glob("*.parquet")
            ).reset_index()
            impressions_dl_path = Path(prefix) / Path("impressions-direct-link")
            self.impressions_dl = pd.concat(
                pd.read_parquet(p) for p in impressions_dl_path.glob("*.parquet")
            ).reset_index()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "implicit":
            if self.use_polars:
                train_data, val_data, test_data = self._prepare_implicit_pl()
            else:
                train_data, val_data, test_data = self._prepare_implicit_pd()
        elif self.data_type == "implicit_bpr":
            train_data, val_data, test_data = self._prepare_implicit_bpr()
        return train_data, val_data, test_data

    def _prepare_implicit_pl(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Select 'clicks' only from all interactions
        interactions = self.interactions.filter(pl.col("interaction_type") == 0)

        impressions_dl = self.impressions_dl.explode("recommended_series_list")

        # Join indirectly positive actions with negative (impressions)
        interactions = interactions.join(
            impressions_dl, on="recommendation_id", how="inner"
        )

        # Mark positive interactions with 1 and negative with 0
        interactions = interactions.with_column(
            pl.when(pl.col("series_id") == pl.col("recommended_series_list"))
            .then(1)
            .otherwise(0)
            .alias("target")
        )
        interactions = interactions.rename(
            {
                "user_id": "user",
                "recommended_series_list": "item",
                "utc_ts_milliseconds": "timestamp",
            }
        )
        interactions = interactions.with_column(
            pl.col("timestamp").cast(pl.Datetime).dt.with_time_unit("ms")
        )
        # Handle (user, item) duplicates
        interactions = interactions.groupby(["user", "item"]).agg(
            [pl.sum("target"), pl.max("timestamp")]
        )
        interactions = interactions.with_column(
            pl.when(pl.col("target") > 0).then(1).otherwise(0).alias("target")
        )
        interactions = interactions.sort("timestamp")
        interactions = interactions.cache()

        # Split data
        train_data = interactions.filter(pl.col("timestamp") < dt.date(2019, 4, 14))
        val_data = interactions.filter(pl.col("timestamp") >= dt.date(2019, 4, 14))

        # Prepare user/item to idx mappers based on train data
        train_user_to_idx = train_data.select(
            [
                pl.col("user").unique(),
                pl.col("user").unique().rank().cast(pl.Int64).alias("user_idx") - 1,
            ]
        )
        train_item_to_idx = train_data.select(
            [
                pl.col("item").unique(),
                pl.col("item").unique().rank().cast(pl.Int64).alias("item_idx") - 1,
            ]
        )

        # Map user/item to idx
        train_data = train_data.join(train_user_to_idx, on="user", how="inner")
        train_data = train_data.join(train_item_to_idx, on="item", how="inner")
        val_data = val_data.join(train_user_to_idx, on="user", how="inner")
        val_data = val_data.join(train_item_to_idx, on="item", how="inner")

        # Select valid columns
        train_data = train_data.select(
            [
                pl.col("user_idx").alias("user"),
                pl.col("item_idx").alias("item"),
                "target",
            ]
        )
        val_data = val_data.select(
            [
                pl.col("user_idx").alias("user"),
                pl.col("item_idx").alias("item"),
                "target",
            ]
        )
        test_data = val_data  # test set == validation set (to change in the future!)

        return train_data.collect(), val_data.collect(), test_data.collect()

    def _prepare_implicit_pd(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

        return train_data, val_data, test_data

    def _prepare_implicit_bpr(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Select 'clicks' only from all interactions
        interactions = self.interactions.filter(pl.col("interaction_type") == 0)

        impressions_dl = self.impressions_dl.explode("recommended_series_list")

        # Join indirectly positive actions with negative (impressions)
        interactions = interactions.join(
            impressions_dl, on="recommendation_id", how="inner"
        )

        # Mark positive interactions with 1 and negative with 0
        interactions = interactions.with_column(
            pl.when(pl.col("series_id") == pl.col("recommended_series_list"))
            .then(1)
            .otherwise(0)
            .alias("target")
        )
        interactions = interactions.rename(
            {
                "user_id": "user",
                "recommended_series_list": "item",
                "utc_ts_milliseconds": "timestamp",
            }
        )
        interactions = interactions.with_column(
            pl.col("timestamp").cast(pl.Datetime).dt.with_time_unit("ms")
        )
        # Handle (user, item) duplicates
        interactions = interactions.groupby(["user", "item"]).agg(
            [pl.sum("target"), pl.max("timestamp")]
        )
        interactions = interactions.with_column(
            pl.when(pl.col("target") > 0).then(1).otherwise(0).alias("target")
        )
        interactions = interactions.sort("timestamp")
        interactions = interactions.cache()

        # Split data
        train_data = interactions.filter(pl.col("timestamp") < dt.date(2019, 4, 14))
        val_data = interactions.filter(pl.col("timestamp") >= dt.date(2019, 4, 14))

        # Transform train data to be BPR specific
        train_data_neg = train_data.filter(pl.col("target") == 0)
        train_data_pos = train_data.filter(pl.col("target") == 1)
        train_data = train_data_neg.join(train_data_pos, on="user", how="inner")

        # Prepare user/item to idx mappers
        train_user_to_idx = train_data.select(
            [
                pl.col("user").unique(),
                pl.col("user").unique().rank().cast(pl.Int64).alias("user_idx") - 1,
            ]
        )
        train_item_to_idx = train_data.select(
            [
                pl.concat((pl.col("item"), pl.col("item_right"))).unique(),
                pl.concat((pl.col("item"), pl.col("item_right")))
                .unique()
                .rank()
                .cast(pl.Int64)
                .alias("item_idx")
                - 1,
            ]
        )

        train_data = train_data.join(train_user_to_idx, on="user", how="inner")
        train_data = train_data.join(train_item_to_idx, on="item", how="inner")
        train_data = train_data.join(
            train_item_to_idx, left_on="item_right", right_on="item", how="inner"
        )
        val_data = val_data.join(train_user_to_idx, on="user", how="inner")
        val_data = val_data.join(train_item_to_idx, on="item", how="inner")

        # Select valid columns
        train_data = train_data.select(
            [
                pl.col("user_idx").alias("user"),
                pl.col("item_idx").alias("item_neg"),
                pl.col("item_idx_right").alias("item_pos"),
            ]
        )
        val_data = val_data.select(
            [
                pl.col("user_idx").alias("user"),
                pl.col("item_idx").alias("item"),
                "target",
            ]
        )
        test_data = val_data  # test set == validation set (to change in the future!)

        return train_data.collect(), val_data.collect(), test_data.collect()

    def save_data(self, train_data, val_data, test_data):
        if self.use_polars:
            train_data.write_parquet(
                f"data/contentwise/train_data_{self.data_type}.parquet"
            )
            val_data.write_parquet(
                f"data/contentwise/val_data_{self.data_type}.parquet"
            )
            test_data.write_parquet(
                f"data/contentwise/test_data_{self.data_type}.parquet"
            )
        else:
            train_data.to_parquet(
                f"data/contentwise/train_data_{self.data_type}.parquet", index=False
            )
            val_data.to_parquet(
                f"data/contentwise/val_data_{self.data_type}.parquet", index=False
            )
            test_data.to_parquet(
                f"data/contentwise/test_data_{self.data_type}.parquet", index=False
            )
