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
    """
    Load raw interaction and impression data from local cache.
    If data is not available locally, download it from remote storage.

    Returns:
        tuple[DataFrame, DataFrame]: DataFrames containing interactions and impressions data.
    """
    data_dir = get_cache_path() / "data-cw10m"
    # Download data if doesn't exist
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


def process_data(
    interactions: DataFrame,
    impressions: DataFrame,
    list_size: int,
    split_date: str,
    user_history_size: int,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Process and clean raw interaction and impression data, splitting it into training and validation sets.

    Args:
        interactions (DataFrame): Raw interactions data.
        impressions (DataFrame): Raw impressions data.
        list_size (int): Minimum list size to be included in the dataset.
        split_date (str): Date string to split train and validation sets (format: YYYY/MM/DD).
        user_history_size (int): Number of past clicks to store as user history.

    Returns:
        tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
            - Processed training dataset.
            - Processed validation dataset.
            - User-to-index mapping DataFrame.
            - Item-to-index mapping DataFrame.
    """
    data = _preprocess_data(interactions, impressions)

    # Remove lists shorter than 'list_size'
    data_valid_lists = (
        data.groupby("list_id").size().reset_index().rename(columns={0: "list_size"})
    )
    data_valid_lists = data_valid_lists[data_valid_lists["list_size"] >= list_size]
    data_valid_lists = data_valid_lists.drop(columns="list_size")
    data = data.merge(data_valid_lists, "inner", "list_id")

    # Split data (train: 2019/01/07-2019/04/08; val: 2019/04/09-2019/04/15)
    split_date = dt.datetime.strptime(split_date, "%Y/%m/%d")
    train = data[data["timestamp"] < split_date]
    val = data[data["timestamp"] >= split_date]

    # Remove overlapping lists between train and val
    train_lists = train["list_id"].unique()
    val_lists = val["list_id"].unique()
    train_lists_expected = np.setdiff1d(train_lists, val_lists)
    val_lists_expected = np.setdiff1d(val_lists, train_lists)
    train = train[train["list_id"].isin(train_lists_expected)]
    val = val[val["list_id"].isin(val_lists_expected)]

    # Remove cold users from val
    train_users = train["user"].unique()
    val = val[val["user"].isin(train_users)]

    # Remove lists containing cold items on val
    train_items = train["item"].unique()
    val_items = val["item"].unique()
    cold_items = np.setdiff1d(val_items, train_items)
    val["cold_item"] = val["item"].isin(cold_items)
    n_cold_items_per_list = val.groupby("list_id")["cold_item"].sum()
    valid_lists = n_cold_items_per_list[n_cold_items_per_list == 0].index
    val = val[val["list_id"].isin(valid_lists)]
    val = val.drop(columns="cold_item")

    # Prepare user/item to idx mappers based on train data
    unique_train_users = train["user"].unique()
    unique_train_items = train["item"].unique()
    user_mapper = pd.DataFrame(
        {"user": unique_train_users, "user_idx": np.arange(unique_train_users.size)}
    )
    item_mapper = pd.DataFrame(
        {"item": unique_train_items, "item_idx": np.arange(unique_train_items.size)}
    )

    # Create user_history column - list of last n clicked items per user
    train_clicks = train.sort_values(by=["user", "timestamp"])
    train_clicks = train_clicks[train_clicks["target"] == 1].reset_index(drop=True)

    def last_clicks(series):
        history = []
        result = []
        for item in series:
            result.append(history.copy())  # Append the current state of history
            if len(history) == user_history_size:
                history.pop(0)  # Keep only the last n items
            history.append(item)
        # Pad with None if history is shorter than threshold
        return [([None] * (user_history_size - len(h)) + h) for h in result]

    # Apply function per user
    train_clicks["user_history"] = train_clicks.groupby("user")["item"].transform(
        last_clicks
    )
    train = train.merge(
        train_clicks[["timestamp", "user", "user_history"]],
        on=["timestamp", "user"],
        how="left",
    )
    tmp = train_clicks.loc[
        train_clicks.groupby("user")["timestamp"].idxmax()
    ].reset_index(drop=True)
    tmp = tmp[["user", "user_history"]]
    val = val.merge(tmp, "inner", "user")

    # Sort train and val by timestamp
    train = train.sort_values("timestamp").reset_index(drop=True)
    val = val.sort_values("timestamp").reset_index(drop=True)

    return train, val, user_mapper, item_mapper


def save_data(
    task: Task,
    train: DataFrame,
    val: DataFrame,
    user_mapper: DataFrame,
    item_mapper: DataFrame,
) -> None:
    """
    Save processed data and mappings as artifacts in ClearML.

    Args:
        task (Task): ClearML task instance.
        train (DataFrame): Processed training dataset.
        val (DataFrame): Processed validation dataset.
        user_mapper (DataFrame): User-to-index mapping.
        item_mapper (DataFrame): Item-to-index mapping.
    """
    task.upload_artifact("train", train, extension_name=".parquet")
    task.upload_artifact("val", val, extension_name=".parquet")
    task.upload_artifact("user_mapper", user_mapper, extension_name=".parquet")
    task.upload_artifact("item_mapper", item_mapper, extension_name=".parquet")


def _download_data(data_dir: Path) -> None:
    """
    Download interaction and impression data from remote S3 storage.

    Args:
        data_dir (Path): Local directory path to store downloaded data.
    """
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


def _preprocess_data(
    interactions: DataFrame,
    impressions_dl: DataFrame,
) -> DataFrame:
    """
    Preprocess raw interactions and impressions data for further processing.
    """
    # Select only movies from item types
    interactions = interactions[interactions["item_type"] == 0]
    # Select only clicks as an interaction type
    interactions = interactions[interactions["interaction_type"] == 0]

    interactions["utc_ts_milliseconds"] = pd.to_datetime(
        interactions["utc_ts_milliseconds"],
        unit="ms",
    )

    # Assume that user can have only one interaction at exact timestamp
    interactions = interactions.drop_duplicates(["utc_ts_milliseconds", "user_id"])

    impressions_dl = impressions_dl.explode("recommended_series_list")
    impressions_dl["recommended_series_list"] = pd.to_numeric(
        impressions_dl["recommended_series_list"]
    )

    # Join positive interactions (clicks) with negative interactions (impressions)
    data = interactions.merge(impressions_dl, "inner", "recommendation_id")

    # Create unique id per (utc_ts_milliseconds, recommandation_id, user_id)
    data["list_id"] = pd.factorize(
        data[["utc_ts_milliseconds", "recommendation_id", "user_id"]].apply(
            tuple, axis=1
        )
    )[0]

    # Mark positive interactions with 1 and negative with 0
    data["target"] = np.where(
        data["series_id"] == data["recommended_series_list"],
        1,
        0,
    )
    data["target"] = data["target"].astype("int32")

    data = data[
        [
            "utc_ts_milliseconds",
            "list_id",
            "user_id",
            "recommended_series_list",
            "target",
        ]
    ]
    data.columns = ["timestamp", "list_id", "user", "item", "target"]

    return data
