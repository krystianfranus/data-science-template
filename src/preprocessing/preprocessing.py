import datetime as dt
from typing import Tuple

import pandas as pd
import polars as pl


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
        self.data = self.data.drop(columns=["timestamp"])

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
        # Split data into train/val/test
        train_data = self.data[:900_000].reset_index(drop=True)
        val_data = self.data[900_000:950_000].reset_index(drop=True)
        test_data = self.data[950_000:].reset_index(drop=True)

        # Prepare unique train user and items
        train_users = train_data["user"].unique()
        train_items = train_data["item"].unique()

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
        train_data["item"] = train_data["item"].map(item_to_idx)
        val_data["user"] = val_data["user"].map(user_to_idx)
        val_data["item"] = val_data["item"].map(item_to_idx)
        test_data["user"] = test_data["user"].map(user_to_idx)
        test_data["item"] = test_data["item"].map(item_to_idx)

        return train_data, val_data, test_data

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data.loc[self.data["target"] < 4, "target"] = 0
        self.data.loc[self.data["target"] >= 4, "target"] = 1

        # Split data into train/val/test
        train_data = self.data[:900_000].reset_index(drop=True)
        val_data = self.data[900_000:950_000].reset_index(drop=True)
        test_data = self.data[950_000:].reset_index(drop=True)

        # Prepare unique train user and items
        train_users = train_data["user"].unique()
        train_items = train_data["item"].unique()

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
        train_data["item"] = train_data["item"].map(item_to_idx)
        val_data["user"] = val_data["user"].map(user_to_idx)
        val_data["item"] = val_data["item"].map(item_to_idx)
        test_data["user"] = test_data["user"].map(user_to_idx)
        test_data["item"] = test_data["item"].map(item_to_idx)

        return train_data, val_data, test_data

    def _prepare_implicit_bpr(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data.loc[self.data["target"] < 4, "target"] = 0
        self.data.loc[self.data["target"] >= 4, "target"] = 1

        # Split data into train/val/test
        train_data = self.data[:900_000].reset_index(drop=True)
        tmp0 = train_data.loc[train_data["target"] == 0, ["user", "item"]]
        tmp1 = train_data.loc[train_data["target"] == 1, ["user", "item"]]
        train_data = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
        train_data = train_data.sample(frac=0.05, random_state=0).reset_index(drop=True)
        val_data = self.data[900_000:950_000].reset_index(drop=True)
        test_data = self.data[950_000:].reset_index(drop=True)

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

        return train_data, val_data, test_data

    def save_data(self, train_data, val_data, test_data):
        train_data.to_csv(
            f"data/movielens/train_data_{self.data_type}.csv", index=False, header=False
        )
        val_data.to_csv(
            f"data/movielens/val_data_{self.data_type}.csv", index=False, header=False
        )
        test_data.to_csv(
            f"data/movielens/test_data_{self.data_type}.csv", index=False, header=False
        )


class ContentWise:
    def __init__(self, remote_data, data_type):
        if remote_data:  # TODO
            prefix = "s3://kfranus-bucket"
            interactions_path = (
                f"{prefix}/data-science-template/data/movielens/ratings.dat"
            )
            impressions_dl_path = (
                f"{prefix}/data-science-template/data/movielens/ratings.dat"
            )
        else:
            prefix = "data/contentwise/data/contentwise/CW10M"
            interactions_path = f"{prefix}/interactions/*.parquet"
            impressions_dl_path = f"{prefix}/impressions-direct-link/*.parquet"

        self.interactions = pl.scan_parquet(interactions_path)
        self.impressions_dl = pl.scan_parquet(impressions_dl_path)
        self.data_type = data_type

    def preprocess(self):
        pass

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
