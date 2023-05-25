import datetime as dt

import numpy as np
import pandas as pd
import s3fs
from clearml import Task
from pandas import DataFrame


def process_data(type: str, task: Task, s3_cfg: dict) -> None:
    params = {}
    interactions, impressions_dl = _load_data(s3_cfg)
    train, val, test = _prepare_data(interactions, impressions_dl, type, params)
    _save_data(train, val, test, params, task)


def _load_data(s3_cfg: dict) -> tuple[DataFrame, DataFrame]:
    s3 = s3fs.S3FileSystem(key=s3_cfg["key"], secret=s3_cfg["secret"])
    bucket_name = "kf-north-bucket"
    prefix = "data-science-template/data/contentwise/CW10M"

    file_paths1 = s3.glob(f"{bucket_name}/{prefix}/interactions/*.parquet")
    dfs = []
    for file_path in file_paths1:
        with s3.open(file_path, "rb") as file:
            df = pd.read_parquet(file)
            dfs.append(df)
    interactions = pd.concat(dfs).reset_index()

    file_paths2 = s3.glob(f"{bucket_name}/{prefix}/impressions-direct-link/*.parquet")
    dfs = []
    for file_path in file_paths2:
        with s3.open(file_path, "rb") as file:
            df = pd.read_parquet(file)
            dfs.append(df)
    impressions_dl = pd.concat(dfs).reset_index()

    return interactions, impressions_dl


def _prepare_data(
    interactions: DataFrame,
    impressions_dl: DataFrame,
    type: str,
    params: dict,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    match type:
        case "simple":
            train, val, test = _prepare_simple(interactions, impressions_dl, params)
        case "bpr":
            train, val, test = _prepare_bpr(interactions, impressions_dl, params)
        case _:
            raise ValueError(f"Invalid data type, you provided '{type}'")

    return train, val, test


def _save_data(
    train: DataFrame, val: DataFrame, test: DataFrame, params: dict, task: Task
) -> None:
    task.connect(params)
    task.upload_artifact("train", train, extension_name=".parquet")
    task.upload_artifact("val", val, extension_name=".parquet")
    task.upload_artifact("test", test, extension_name=".parquet")


def _prepare_simple(
    interactions: DataFrame,
    impressions_dl: DataFrame,
    params: dict,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    interactions = _common(interactions, impressions_dl)

    # Split data
    train = interactions[interactions["timestamp"] < dt.datetime(2019, 4, 14)]
    val = interactions[interactions["timestamp"] >= dt.datetime(2019, 4, 14)]

    # Prepare user/item to idx mappers based on train data
    unique_users = train["user"].unique()
    unique_items = train["item"].unique()
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

    return train, val, test


def _prepare_bpr(
    interactions: DataFrame,
    impressions_dl: DataFrame,
    params: dict,
) -> tuple[DataFrame, DataFrame, DataFrame]:
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

    return train, val, test


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
