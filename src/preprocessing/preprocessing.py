import datetime as dt
from typing import Tuple

import pandas as pd
import polars as pl


class ContentWise:
    def __init__(self, data_type, use_remote_storage):
        if use_remote_storage:
            prefix = "s3://kf-north-bucket/data-science-template/data/contentwise/CW10M"
        else:
            prefix = "data/contentwise/data/contentwise/CW10M"
        interactions_path = f"{prefix}/interactions/*.parquet"
        impressions_dl_path = f"{prefix}/impressions-direct-link/*.parquet"

        self.interactions = pl.scan_parquet(interactions_path)
        self.impressions_dl = pl.scan_parquet(impressions_dl_path)
        self.data_type = data_type

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "implicit":
            train_data, val_data, test_data = self._prepare_implicit()
        elif self.data_type == "implicit_bpr":
            train_data, val_data, test_data = self._prepare_implicit_bpr()
        return train_data, val_data, test_data

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        train_data.write_parquet(
            f"data/contentwise/train_data_{self.data_type}.parquet"
        )
        val_data.write_parquet(f"data/contentwise/val_data_{self.data_type}.parquet")
        test_data.write_parquet(f"data/contentwise/test_data_{self.data_type}.parquet")
