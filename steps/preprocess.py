import logging

import hydra
from clearml import Task, TaskTypes
from omegaconf import DictConfig

from mypackage.preprocessing.preprocessing import ContentWise

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs/preprocessing/",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    output_uri = None
    if cfg.use_remote_storage:
        output_uri = "s3://kf-north-bucket/data-science-template/output/"
    task = Task.init(
        project_name="MyProject",
        task_name="Preprocessing",
        task_type=TaskTypes.data_processing,
        reuse_last_task_id=False,
        output_uri=output_uri,
    )
    if cfg.draft_mode:
        task.execute_remotely()

    log.info("Data loading")
    data = ContentWise(cfg.data_type)

    log.info("Data parsing")
    data.prepare_data()

    log.info("Data saving")
    task.connect({"n_users": data.n_users, "n_items": data.n_items})
    task.upload_artifact("train_data", data.train_data, extension_name=".parquet")
    task.upload_artifact("val_data", data.val_data, extension_name=".parquet")
    task.upload_artifact("test_data", data.test_data, extension_name=".parquet")

    log.info("Done!")


if __name__ == "__main__":
    main()
