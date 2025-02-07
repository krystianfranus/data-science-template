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


def load_raw_data() -> tuple[DataFrame, DataFrame]:
    data_dir = get_cache_path() / "data-cw10m"
    # Download data if necessary
    if not data_dir.exists():
        log.info("Data are not available locally!")
        log.info(f"Downloading data from remote storage to {data_dir} ...")
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


def process_simple(
    interactions: DataFrame,
    impressions: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    interactions = _common(interactions, impressions)

    # Split data (train: 2019/01/07-2019/04/13; val: 2019/04/14-2019/04/15)
    split_date = dt.datetime(2019, 4, 14)
    train = interactions[interactions["timestamp"] < split_date]
    val = interactions[interactions["timestamp"] >= split_date]

    # Keep lists with condition 0 < mean_target < 1
    train_valid_lists = (
        train.groupby("list_id")
        .agg({"target": "mean"})
        .rename(columns={"target": "mean"})
        .reset_index()
    )
    train_valid_lists = train_valid_lists[
        (train_valid_lists["mean"] > 0) & (train_valid_lists["mean"] < 1)
    ]
    train = train.merge(train_valid_lists, "inner", "list_id")

    # Prepare user/item to idx mappers based on train data
    unique_train_users = train["user"].unique()
    unique_train_items = train["item"].unique()
    user_mapper = pd.DataFrame(
        {"user": unique_train_users, "user_idx": np.arange(unique_train_users.size)}
    )
    item_mapper = pd.DataFrame(
        {"item": unique_train_items, "item_idx": np.arange(unique_train_items.size)}
    )

    # Map user/item to idx - it removes cold users and items from validation
    train = train.merge(user_mapper, on="user", how="inner")
    train = train.merge(item_mapper, on="item", how="inner")
    val = val.merge(user_mapper, on="user", how="inner")
    val = val.merge(item_mapper, on="item", how="inner")

    # Keep lists with condition 0 < mean_target < 1
    val_valid_lists = (
        val.groupby("list_id")
        .agg({"target": "mean"})
        .rename(columns={"target": "mean"})
        .reset_index()
    )
    val_valid_lists = val_valid_lists[
        (val_valid_lists["mean"] > 0) & (val_valid_lists["mean"] < 1)
    ]
    val = val.merge(val_valid_lists, "inner", "list_id")

    train = train.sort_values("timestamp").reset_index(drop=True)
    val = val.sort_values("timestamp").reset_index(drop=True)

    # Select valid columns
    train = train[["timestamp", "list_id", "user_idx", "item_idx", "target"]]
    train.columns = ["timestamp", "list_id", "user", "item", "target"]
    val = val[["timestamp", "list_id", "user_idx", "item_idx", "target"]]
    val.columns = ["timestamp", "list_id", "user", "item", "target"]

    # Mock test_data
    test = val.copy()  # test set == validation set (should be changed in the future!)

    # Prepare statistics
    unique_val_users = val["user"].unique()
    unique_val_items = val["item"].unique()
    stats = {}
    stats["train_n_users"] = unique_train_users.size
    stats["train_n_items"] = unique_train_items.size
    stats["train_n_lists"] = train["list_id"].nunique()
    stats["train_n_clicks"] = int(train["target"].sum())
    stats["train_n_impressions"] = len(train) - stats["train_n_clicks"]
    stats["train_ctr"] = stats["train_n_clicks"] / stats["train_n_impressions"]
    stats["val_n_users"] = unique_val_users.size
    stats["val_n_items"] = unique_val_items.size
    stats["val_n_lists"] = val["list_id"].nunique()
    stats["val_n_clicks"] = int(val["target"].sum())
    stats["val_n_impressions"] = len(val) - stats["val_n_clicks"]
    stats["val_ctr"] = stats["val_n_clicks"] / stats["val_n_impressions"]

    return train, val, test, stats, user_mapper, item_mapper


def process_bpr(
    interactions: DataFrame,
    impressions: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    interactions = _common(interactions, impressions)

    # Split data
    split_date = dt.datetime(2019, 4, 14)
    train = interactions[interactions["timestamp"] < split_date]
    tmp0 = train.loc[train["target"] == 0, ["user", "item"]]
    tmp1 = train.loc[train["target"] == 1, ["user", "item"]]
    train = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
    val = interactions[interactions["timestamp"] >= split_date]

    # Prepare user/item to idx mappers based on train data
    unique_train_users = train["user"].unique()
    # unique_users = train["user"].unique()
    item_neg_set = set(train["item_neg"])
    item_pos_set = set(train["item_pos"])
    unique_train_items = pd.Series(list(item_neg_set | item_pos_set)).unique()

    user_mapper = pd.DataFrame(
        {"user": unique_train_users, "user_idx": np.arange(unique_train_users.size)}
    )
    item_mapper = pd.DataFrame(
        {"item": unique_train_items, "item_idx": np.arange(unique_train_items.size)}
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

    # Keep lists with condition 0 < mean_target < 1
    val_valid_lists = (
        val.groupby("list_id")
        .agg({"target": "mean"})
        .rename(columns={"target": "mean"})
        .reset_index()
    )
    val_valid_lists = val_valid_lists[
        (val_valid_lists["mean"] > 0) & (val_valid_lists["mean"] < 1)
    ]
    val = val.merge(val_valid_lists, "inner", "list_id")

    val = val[["timestamp", "list_id", "user_idx", "item_idx", "target"]]
    val = val.rename(columns={"user_idx": "user", "item_idx": "item"})

    # Mock test_data
    test = val.copy()  # test set == validation set (to change in the future!)

    # Prepare statistics
    unique_val_users = val["user"].unique()
    unique_val_items = val["item"].unique()
    stats = {}
    stats["train_n_users"] = unique_train_users.size
    stats["train_n_items"] = unique_train_items.size
    stats["val_n_users"] = unique_val_users.size
    stats["val_n_items"] = unique_val_items.size
    stats["val_n_lists"] = val["list_id"].nunique()
    stats["val_n_clicks"] = int(val["target"].sum())
    stats["val_n_impressions"] = len(val) - stats["val_n_clicks"]
    stats["val_ctr"] = stats["val_n_clicks"] / stats["val_n_impressions"]

    return train, val, test, stats, user_mapper, item_mapper


def save_data(
    task: Task,
    train: DataFrame,
    val: DataFrame,
    test: DataFrame,
    stats: dict,
    user_mapper: DataFrame,
    item_mapper: DataFrame,
) -> None:
    task.upload_artifact("train", train, extension_name=".parquet")
    task.upload_artifact("val", val, extension_name=".parquet")
    task.upload_artifact("test", test, extension_name=".parquet")
    task.upload_artifact("stats", stats)
    task.upload_artifact("user_mapper", user_mapper, extension_name=".parquet")
    task.upload_artifact("item_mapper", item_mapper, extension_name=".parquet")


def _download_data(data_dir: Path) -> None:
    s3 = s3fs.S3FileSystem()
    prefix = "s3://kf-north-bucket/data-science-template/data/contentwise/CW10M"

    # Downloading parquets with interactions
    (data_dir / "interactions").mkdir(parents=True, exist_ok=False)
    interactions_s3_paths = s3.glob(f"{prefix}/interactions/*.parquet")
    for p in interactions_s3_paths:
        with s3.open(p, "rb") as file:
            df = pd.read_parquet(file)
            df.to_parquet(f"{data_dir}/interactions/{Path(p).name}")

    # Downloading parquets with impressions
    (data_dir / "impressions-direct-link").mkdir(parents=True, exist_ok=False)
    impressions_s3_paths = s3.glob(f"{prefix}/impressions-direct-link/*.parquet")
    for p in impressions_s3_paths:
        with s3.open(p, "rb") as file:
            df = pd.read_parquet(file)
            df.to_parquet(f"{data_dir}/impressions-direct-link/{Path(p).name}")


def _common(
    interactions: DataFrame,
    impressions_dl: DataFrame,
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

    # Create unique id per (recommandation_id, user_id) pairs
    interactions["list_id"] = pd.factorize(
        interactions[["recommendation_id", "user_id"]].apply(tuple, axis=1)
    )[0]

    # Mark positive interactions with 1 and negative with 0
    interactions["target"] = np.where(
        interactions["series_id"] == interactions["recommended_series_list"],
        1,
        0,
    )
    interactions["target"] = interactions["target"].astype("int32")

    interactions = interactions[
        [
            "utc_ts_milliseconds",
            "list_id",
            "user_id",
            "recommended_series_list",
            "target",
        ]
    ]
    interactions.columns = ["timestamp", "list_id", "user", "item", "target"]

    return interactions
