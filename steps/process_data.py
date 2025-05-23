import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.data_processing.data_processing import (
    load_raw_data,
    process_data,
    save_data,
)

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "data_processing"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    task = Task.init(
        project_name=cfg.project_name,
        task_name="DataProcessing",
        task_type=TaskTypes.data_processing,
        tags=cfg.tags,
        reuse_last_task_id=False,
    )

    log.info("Loading raw (CW10M) data")
    interactions, impressions = load_raw_data()
    log.info("Raw data (CW10M) loaded successfully!")

    log.info("Processing raw data")
    train, val, user_mapper, item_mapper, last_user_histories = process_data(
        interactions,
        impressions,
        cfg.list_size,
        cfg.split_date,
        cfg.user_history_size,
    )
    log.info("Raw data processed successfully!")

    log.info("Saving processed data")
    save_data(task, train, val, user_mapper, item_mapper, last_user_histories)
    log.info("Processed data saved successfully!")


if __name__ == "__main__":
    main()
