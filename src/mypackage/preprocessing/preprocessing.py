import datetime as dt
import logging
import os
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
    cache_dir = get_cache_path() / "CW10M"
    if cache_dir.exists():
        paths1 = glob(os.path.join(cache_dir, "interactions", "*.parquet"))
        interactions = pd.concat([pd.read_parquet(p) for p in paths1]).reset_index()
        paths2 = glob(os.path.join(cache_dir, "impressions-direct-link", "*.parquet"))
        impressions_dl = pd.concat([pd.read_parquet(p) for p in paths2]).reset_index()
    else:
        log.info("Data are not available locally (downloading from remote storage)")

        (cache_dir / "interactions").mkdir(parents=True, exist_ok=False)
        (cache_dir / "impressions-direct-link").mkdir(parents=True, exist_ok=False)
        s3_prefix = "kf-north-bucket/data-science-template/data/contentwise/CW10M"

        key = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        s3 = s3fs.S3FileSystem(key=key, secret=secret)

        paths1 = s3.glob(os.path.join(s3_prefix, "interactions", "*.parquet"))
        dfs = []
        for p in paths1:
            with s3.open(p, "rb") as file:
                df = pd.read_parquet(file)
                df.to_parquet(f"cache/CW10M/interactions/{Path(p).name}")
                dfs.append(df)
        interactions = pd.concat(dfs).reset_index()

        paths2 = s3.glob(f"{s3_prefix}/impressions-direct-link/*.parquet")
        dfs = []
        for p in paths2:
            with s3.open(p, "rb") as file:
                df = pd.read_parquet(file)
                df.to_parquet(f"cache/CW10M/impressions-direct-link/{Path(p).name}")
                dfs.append(df)
        impressions_dl = pd.concat(dfs).reset_index()

    return interactions, impressions_dl


def prepare_data(
    interactions: DataFrame,
    impressions_dl: DataFrame,
    type: str,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    match type:
        case "simple":
            train, val, test, params = _prepare_simple(interactions, impressions_dl)
        case "bpr":
            train, val, test, params = _prepare_bpr(interactions, impressions_dl)
        case _:
            raise ValueError(f"Invalid data type, you provided '{type}'")

    return train, val, test, params


def save_data(
    task: Task, train: DataFrame, val: DataFrame, test: DataFrame, params: dict
) -> None:
    task.connect(params)
    task.upload_artifact("train", train, extension_name=".parquet")
    task.upload_artifact("val", val, extension_name=".parquet")
    task.upload_artifact("test", test, extension_name=".parquet")


def _prepare_simple(
    interactions: DataFrame,
    impressions_dl: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    interactions = _common(interactions, impressions_dl)

    # Split data
    train = interactions[interactions["timestamp"] < dt.datetime(2019, 4, 14)]
    val = interactions[interactions["timestamp"] >= dt.datetime(2019, 4, 14)]

    # Prepare user/item to idx mappers based on train data
    unique_users = train["user"].unique()
    unique_items = train["item"].unique()
    params = {}
    params["n_users"] = unique_users.size
    params["n_items"] = unique_items.size
    params["n_clicks"] = int(train["target"].sum())
    params["n_impressions"] = len(train) - params["n_clicks"]
    train_user_to_idx = pd.DataFrame(
        {"user": unique_users, "user_idx": np.arange(params["n_users"])}
    )
    train_item_to_idx = pd.DataFrame(
        {"item": unique_items, "item_idx": np.arange(params["n_items"])}
    )

    # Map user/item to idx
    train = train.merge(train_user_to_idx, on="user", how="inner")
    train = train.merge(train_item_to_idx, on="item", how="inner")
    val = val.merge(train_user_to_idx, on="user", how="inner")
    val = val.merge(train_item_to_idx, on="item", how="inner")

    train = train.sort_values("timestamp").reset_index(drop=True)
    val = val.sort_values("timestamp").reset_index(drop=True)

    # Select valid columns
    train = train[["user_idx", "item_idx", "target"]]
    train.columns = ["user", "item", "target"]
    val = val[["user_idx", "item_idx", "target"]]
    val.columns = ["user", "item", "target"]

    # Mock test_data
    test = val.copy()  # test set == validation set (to change in the future!)

    return train, val, test, params


def _prepare_bpr(
    interactions: DataFrame,
    impressions_dl: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, dict]:
    interactions = _common(interactions, impressions_dl)

    # Split data
    train = interactions[interactions["timestamp"] < dt.datetime(2019, 4, 14)]
    tmp0 = train.loc[train["target"] == 0, ["user", "item"]]
    tmp1 = train.loc[train["target"] == 1, ["user", "item"]]
    train = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
    val = interactions[interactions["timestamp"] >= dt.datetime(2019, 4, 14)]

    # Prepare user/item to idx mappers based on train data
    unique_users = train["user"].unique()
    item_neg_set = set(train["item_neg"])
    item_pos_set = set(train["item_pos"])
    unique_items = pd.Series(list(item_neg_set | item_pos_set)).unique()
    params = {}
    params["n_users"] = unique_users.size
    params["n_items"] = unique_items.size
    train_user_to_idx = pd.DataFrame(
        {"user": unique_users, "user_idx": np.arange(params["n_users"])}
    )
    train_item_to_idx = pd.DataFrame(
        {"item": unique_items, "item_idx": np.arange(params["n_items"])}
    )

    # Map user/item to idx and handle column names conflicts
    train = train.merge(train_user_to_idx, on="user", how="inner")
    train = train[["user_idx", "item_neg", "item_pos"]].rename(
        columns={"user_idx": "user"}
    )
    train = train.merge(
        train_item_to_idx, left_on="item_neg", right_on="item", how="inner"
    )
    train = train[["user", "item_idx", "item_pos"]].rename(
        columns={"item_idx": "item_neg"}
    )
    train = train.merge(
        train_item_to_idx, left_on="item_pos", right_on="item", how="inner"
    )
    train = train[["user", "item_neg", "item_idx"]].rename(
        columns={"item_idx": "item_pos"}
    )

    val = val.merge(train_user_to_idx, on="user", how="inner")
    val = val.merge(train_item_to_idx, on="item", how="inner")
    val = val[["user_idx", "item_idx", "target"]].rename(
        columns={"user_idx": "user", "item_idx": "item"}
    )

    # Mock test_data
    test = val.copy()  # test set == validation set (to change in the future!)

    return train, val, test, params


def _common(interactions: DataFrame, impressions_dl: DataFrame) -> DataFrame:
    # Select movies only from other item types
    interactions = interactions[interactions["item_type"] == 0]
    # Select clicks only from other interaction types
    interactions = interactions[interactions["interaction_type"] == 0]

    interactions["utc_ts_milliseconds"] = pd.to_datetime(
        interactions["utc_ts_milliseconds"], unit="ms"
    )

    impressions_dl = impressions_dl.explode("recommended_series_list")
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
