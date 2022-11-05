import logging

import hydra
from clearml import Task, TaskTypes
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/preprocessing/", config_name="config"
)
def main(config: DictConfig):
    if config.execute_locally:
        remote_data = False
    else:
        task = Task.init(
            project_name="My project",
            task_name="Preprocessing",
            task_type=TaskTypes.data_processing,
            output_uri="s3://kfranus-bucket/data-science-template/output/",
        )
        remote_data = True

        if config.draft_mode:
            task.execute_remotely()

    data = hydra.utils.instantiate(config.data, remote_data=remote_data)
    log.info("[My Logger] Data loading")
    data.preprocess()
    log.info("[My Logger] Data parsing")
    train_data, val_data, test_data = data.prepare_data()

    log.info("[My Logger] Data (artifacts) uploading")
    if config.execute_locally:
        data.save_data(train_data, val_data, test_data)
    else:
        task.upload_artifact("train_data", train_data)
        task.upload_artifact("val_data", val_data)
        task.upload_artifact("test_data", test_data)

    log.info("[My Logger] Done!")


if __name__ == "__main__":
    main()
