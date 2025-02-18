import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.data_processing.data_processing import load_raw_data, save_data

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "data_processing"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    data_processing_method_name = cfg.method._target_.split(".")[-1]

    output_uri = None
    if cfg.use_remote_storage:
        output_uri = "s3://kf-north-bucket/data-science-template/output/"

    task = Task.init(
        project_name=cfg.project_name,
        task_name="DataProcessing",
        task_type=TaskTypes.data_processing,
        tags=[data_processing_method_name],
        reuse_last_task_id=False,
        output_uri=output_uri,
    )

    log.info("Loading raw (CW10M) data")
    interactions, impressions = load_raw_data()
    log.info("Raw data (CW10M) loaded successfully!")

    log.info(f"Processing raw data using '{data_processing_method_name}' method")
    train, val, test, stats, user_mapper, item_mapper = hydra.utils.call(
        cfg.method,
        interactions=interactions,
        impressions=impressions,
    )
    log.info("Raw data processed successfully!")

    log.info("Saving processed data")
    save_data(task, train, val, test, stats, user_mapper, item_mapper)
    log.info("Processed data saved successfully!")


if __name__ == "__main__":
    main()
