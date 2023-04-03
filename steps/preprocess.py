import logging

import hydra
from clearml import Task, TaskTypes
from omegaconf import DictConfig

from src.preprocessing.preprocessing import ContentWise

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/preprocessing/", config_name="config"
)
def main(cfg: DictConfig):

    if cfg.clearml:
        output_uri = None
        if cfg.use_remote_storage:
            output_uri = "s3://kf-north-bucket/data-science-template/output/"
        task = Task.init(
            project_name="MyProject",
            task_name="Preprocessing",
            task_type=TaskTypes.data_processing,
            output_uri=output_uri,
        )
        if cfg.draft_mode:
            task.execute_remotely()

    log.info("Data loading")
    data = ContentWise(cfg.data_type)

    log.info("Data parsing")
    train_data, val_data, test_data = data.prepare_data()

    log.info("Data (artifacts) saving")
    if cfg.clearml:
        task.upload_artifact("train_data", train_data, extension_name=".parquet")
        task.upload_artifact("val_data", val_data, extension_name=".parquet")
        task.upload_artifact("test_data", test_data, extension_name=".parquet")
    else:
        data.save_data(train_data, val_data, test_data)


if __name__ == "__main__":
    main()
