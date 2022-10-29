import logging

import hydra
import pandas as pd
from clearml import Task, TaskTypes
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/preprocessing/", config_name="config"
)
def main(config: DictConfig):
    task = Task.init(
        project_name="My project",
        task_name="Preprocessing",
        task_type=TaskTypes.data_processing,
        output_uri="s3://kfranus-bucket/data",
    )

    # if config.execute_remotely:
    #     task.execute_remotely()

    log.info("[My Logger] Data loading")
    raw_data_path = config.raw_data_path
    ratings = pd.read_csv(
        raw_data_path,
        sep="::",
        names=["user", "item", "rating", "timestamp"],
        engine="python",
    )

    log.info("[My Logger] Data splitting")
    train_data = ratings[:700_000].reset_index(drop=True)
    val_data = ratings[700_000:850_000].reset_index(drop=True)
    test_data = ratings[850_000:].reset_index(drop=True)
    data = pd.concat((train_data, val_data, test_data)).reset_index(drop=True)

    user_to_idx = {user: idx for idx, user in enumerate(data["user"].unique())}
    item_to_idx = {item: idx for idx, item in enumerate(data["item"].unique())}

    train_data["user"] = train_data["user"].map(user_to_idx)
    val_data["user"] = val_data["user"].map(user_to_idx)
    test_data["user"] = test_data["user"].map(user_to_idx)
    train_data["item"] = train_data["item"].map(item_to_idx)
    val_data["item"] = val_data["item"].map(item_to_idx)
    test_data["item"] = test_data["item"].map(item_to_idx)

    log.info("[My Logger] Data (artifacts) uploading")
    task.upload_artifact("train_data", train_data)
    task.upload_artifact("val_data", val_data)
    task.upload_artifact("test_data", test_data)

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
