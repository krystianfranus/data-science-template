import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    Task.init(
        project_name="MyProject",
        task_name="Baselines",
        task_type=TaskTypes.custom,
        reuse_last_task_id=False,
    )

    task_data_processing = Task.get_task(
        task_id=cfg.baselines.data_processing_clearml_task_id,
        project_name="MyProject",
        task_name="DataProcessing",
    )

    log.info("Loading data")
    train = task_data_processing.artifacts["train"].get()
    val = task_data_processing.artifacts["val"].get()
    log.info("Data loaded successfully!")

    log.info("Computing baselines")
    hydra.utils.call(cfg.baselines.method, train=train, val=val)
    log.info("Baselines computed successfully!")


if __name__ == "__main__":
    main()
