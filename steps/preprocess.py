import logging
import os

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.preprocessing.preprocessing import load_data, prepare_data, save_data

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(get_project_root(), "configs", "preprocessing"),
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
    # if cfg.draft_mode:
    #     task.execute_remotely()

    log.info("Loading raw data")
    interactions, impressions_dl = load_data()

    log.info("Parsing")
    train, val, test, params = prepare_data(interactions, impressions_dl, cfg.type)
    log.info(f"Found data properties: {params}")

    log.info("Saving parsed data")
    save_data(task, train, val, test, params)

    log.info("Done!")


if __name__ == "__main__":
    main()
