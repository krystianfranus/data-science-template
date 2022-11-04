import logging

import hydra
from clearml import Task, TaskTypes
from omegaconf import DictConfig

from src.preprocessing.preprocessing import MovieLens1M

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/preprocessing/", config_name="config"
)
def main(config: DictConfig):
    if config.execute_locally:
        raw_data_path = "data/tmp/movielens/ratings.dat"
    else:
        task = Task.init(
            project_name="My project",
            task_name="Preprocessing",
            task_type=TaskTypes.data_processing,
            output_uri="s3://kfranus-bucket/data-science-template/output/",
        )
        raw_data_path = (
            "s3://kfranus-bucket/data-science-template/data/movielens/ratings.dat"
        )

        if config.draft_mode:
            task.execute_remotely()

    movielens = MovieLens1M(raw_data_path, config.data_type)
    log.info("[My Logger] Data loading")
    movielens.preprocess()
    log.info("[My Logger] Data parsing")
    train_data, val_data, test_data = movielens.prepare_data()

    log.info("[My Logger] Data (artifacts) uploading")
    if config.execute_locally:
        train_data.to_csv(
            f"data/tmp/movielens/train_data_{config.data_type}.csv", index=False
        )
        val_data.to_csv(
            f"data/tmp/movielens/val_data_{config.data_type}.csv", index=False
        )
        test_data.to_csv(
            f"data/tmp/movielens/test_data_{config.data_type}.csv", index=False
        )
    else:
        task.upload_artifact("train_data", train_data)
        task.upload_artifact("val_data", val_data)
        task.upload_artifact("test_data", test_data)

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
