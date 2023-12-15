import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.preprocessing.preprocessing import load_data, prepare_data, save_data

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "preprocessing"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
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

    log.info("Step 1 - Loading raw data")
    interactions, impressions = load_data()
    log.info("Step 1 - Success!")

    log.info("Step 2 - Parsing data")
    train, val, test, stats = prepare_data(
        interactions,
        impressions,
        cfg.data_type,
        cfg.n_user_clicks,
        cfg.n_item_clicks,
    )
    log.info("Step 2 - Success!")
    log.info(f"Details of obtained data: {stats}")

    log.info("Step 3 - Saving prepared data")
    save_data(task, train, val, test, stats)
    log.info("Step 3 - Success!")


if __name__ == "__main__":
    main()
