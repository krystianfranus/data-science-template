import datetime as dt
import logging
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
from clearml import Task
from pandas import DataFrame

from mypackage import get_cache_path

log = logging.getLogger(__name__)


def load_data() -> tuple[DataFrame, DataFrame]:
    data_dir = get_cache_path() / "CW10M"

    # Download data if necessary
    if not data_dir.exists():
        log.info("Data are not available locally!")
        log.info("Downloading data from remote storage to cache ...")
        _download_data(data_dir)

    # Load data containing interactions and impressions
    interactions_paths = glob(f"{data_dir}/interactions/*.parquet")
    interactions = pd.concat(
        [pd.read_parquet(p) for p in interactions_paths],
    ).reset_index()

    impressions_paths = glob(f"{data_dir}/impressions-direct-link/*.parquet")
    impressions = pd.concat(
        [pd.read_parquet(p) for p in impressions_paths],
    ).reset_index()

    return interactions, impressions


def prepare_data(
    interactions: DataFrame,
    impressions: DataFrame,
    data_type: str,
    n_user_clicks: int,
    n_item_clicks: int,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    interactions = _common(interactions, impressions, n_user_clicks, n_item_clicks)

    match data_type:
        case "simple":
            train, val, test, stats = _prepare_simple(interactions)
        case "bpr":
            train, val, test, stats = _prepare_bpr(interactions)
        case _:
            raise ValueError(f"Invalid data type, you provided '{data_type}'")

    return train, val, test, stats


def save_data(
    task: Task, train: DataFrame, val: DataFrame, test: DataFrame, stats: dict
) -> None:
    task.upload_artifact("train", train, extension_name=".parquet")
    task.upload_artifact("val", val, extension_name=".parquet")
    task.upload_artifact("test", test, extension_name=".parquet")
    task.upload_artifact("stats", stats)


def _download_data(data_dir: Path) -> None:
    s3 = s3fs.S3FileSystem()
    prefix = "s3://kf-north-bucket/data-science-template/data/contentwise/CW10M"

    # Saving parquets with interactions
    (data_dir / "interactions").mkdir(parents=True, exist_ok=False)
    interactions_s3_paths = s3.glob(f"{prefix}/interactions/*.parquet")
    for p in interactions_s3_paths:
        with s3.open(p, "rb") as file:
            df = pd.read_parquet(file)
            df.to_parquet(f"{data_dir}/interactions/{Path(p).name}")

    # Saving parquets with impressions
    (data_dir / "impressions-direct-link").mkdir(parents=True, exist_ok=False)
    impressions_s3_paths = s3.glob(f"{prefix}/impressions-direct-link/*.parquet")
    for p in impressions_s3_paths:
        with s3.open(p, "rb") as file:
            df = pd.read_parquet(file)
            df.to_parquet(f"{data_dir}/impressions-direct-link/{Path(p).name}")


def _common(
    interactions: DataFrame,
    impressions_dl: DataFrame,
    n_user_clicks: int,
    n_item_clicks: int,
) -> DataFrame:
    # Select only movies from item types
    interactions = interactions[interactions["item_type"] == 0]
    # Select only clicks as an interaction type
    interactions = interactions[interactions["interaction_type"] == 0]

    interactions["utc_ts_milliseconds"] = pd.to_datetime(
        interactions["utc_ts_milliseconds"],
        unit="ms",
    )

    impressions_dl = impressions_dl.explode("recommended_series_list")
    impressions_dl["recommended_series_list"] = pd.to_numeric(
        impressions_dl["recommended_series_list"]
    )

    # Join positive interactions (clicks) with negative interactions (impressions)
    interactions = interactions.merge(impressions_dl, "inner", "recommendation_id")

    # Mark positive interactions with 1 and negative with 0
    interactions["target"] = np.where(
        interactions["series_id"] == interactions["recommended_series_list"],
        1,
        0,
    )
    interactions["target"] = interactions["target"].astype("int32")

    interactions = interactions[
        ["user_id", "recommended_series_list", "target", "utc_ts_milliseconds"]
    ]
    interactions.columns = ["user", "item", "target", "timestamp"]

    # Handle (user, item) duplicates - keep positive interactions first (if exist)
    interactions = (
        interactions.groupby(["user", "item"])
        .agg({"target": "sum", "timestamp": "max"})
        .reset_index()
    )

    interactions.loc[interactions["target"] > 0, "target"] = 1

    # Get rid of inactive users and items based on specified number of positive
    # interactions
    active_users = (
        interactions.groupby("user")
        .agg({"target": "sum"})
        .rename(columns={"target": "sum"})
        .reset_index()
    )
    active_users = active_users[active_users["sum"] >= n_user_clicks]
    interactions = interactions.merge(active_users, "inner", "user")
    active_items = (
        interactions.groupby("item")
        .agg({"target": "sum"})
        .rename(columns={"target": "sum"})
        .reset_index()
    )
    active_items = active_items[active_items["sum"] >= n_item_clicks]
    interactions = interactions.merge(active_items, "inner", "item")
    interactions = interactions[["user", "item", "target", "timestamp"]]

    return interactions


def _prepare_simple(
    interactions: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    # Split data
    split_date = dt.datetime(2019, 4, 14)
    train = interactions[interactions["timestamp"] < split_date]
    val = interactions[interactions["timestamp"] >= split_date]

    # Prepare user/item to idx mappers based on train data
    unique_users = train["user"].unique()
    unique_items = train["item"].unique()
    stats = {}
    stats["n_users"] = unique_users.size
    stats["n_items"] = unique_items.size
    stats["n_clicks"] = int(train["target"].sum())
    stats["n_impressions"] = len(train) - stats["n_clicks"]
    user_mapper = pd.DataFrame(
        {"user": unique_users, "user_idx": np.arange(stats["n_users"])}
    )
    item_mapper = pd.DataFrame(
        {"item": unique_items, "item_idx": np.arange(stats["n_items"])}
    )

    # Map user/item to idx
    train = train.merge(user_mapper, on="user", how="inner")
    train = train.merge(item_mapper, on="item", how="inner")
    val = val.merge(user_mapper, on="user", how="inner")
    val = val.merge(item_mapper, on="item", how="inner")

    train = train.sort_values("timestamp").reset_index(drop=True)
    val = val.sort_values("timestamp").reset_index(drop=True)

    # Select valid columns
    train = train[["user_idx", "item_idx", "target"]]
    train.columns = ["user", "item", "target"]
    val = val[["user_idx", "item_idx", "target"]]
    val.columns = ["user", "item", "target"]

    # Mock test_data
    test = val.copy()  # test set == validation set (to change in the future!)

    return train, val, test, stats


def _prepare_bpr(
    interactions: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    # Split data
    split_date = dt.datetime(2019, 4, 14)
    train = interactions[interactions["timestamp"] < split_date]
    tmp0 = train.loc[train["target"] == 0, ["user", "item"]]
    tmp1 = train.loc[train["target"] == 1, ["user", "item"]]
    train = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
    val = interactions[interactions["timestamp"] >= split_date]

    # Prepare user/item to idx mappers based on train data
    unique_users = train["user"].unique()
    item_neg_set = set(train["item_neg"])
    item_pos_set = set(train["item_pos"])
    unique_items = pd.Series(list(item_neg_set | item_pos_set)).unique()
    stats = {}
    stats["n_users"] = unique_users.size
    stats["n_items"] = unique_items.size
    user_mapper = pd.DataFrame(
        {"user": unique_users, "user_idx": np.arange(stats["n_users"])}
    )
    item_mapper = pd.DataFrame(
        {"item": unique_items, "item_idx": np.arange(stats["n_items"])}
    )

    # Map user/item to idx and handle column names conflicts
    train = train.merge(user_mapper, on="user", how="inner")
    train = train[["user_idx", "item_neg", "item_pos"]].rename(
        columns={"user_idx": "user"}
    )
    train = train.merge(item_mapper, left_on="item_neg", right_on="item", how="inner")
    train = train[["user", "item_idx", "item_pos"]].rename(
        columns={"item_idx": "item_neg"}
    )
    train = train.merge(item_mapper, left_on="item_pos", right_on="item", how="inner")
    train = train[["user", "item_neg", "item_idx"]].rename(
        columns={"item_idx": "item_pos"}
    )

    val = val.merge(user_mapper, on="user", how="inner")
    val = val.merge(item_mapper, on="item", how="inner")
    val = val[["user_idx", "item_idx", "target"]].rename(
        columns={"user_idx": "user", "item_idx": "item"}
    )

    # Mock test_data
    test = val.copy()  # test set == validation set (to change in the future!)

    return train, val, test, stats
