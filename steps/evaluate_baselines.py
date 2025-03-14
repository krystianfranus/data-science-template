import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.baselines.baselines import evaluate_baselines

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "baselines"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    task_data_processing = Task.get_task(
        task_id=cfg.data_processing_task_id,
        project_name=cfg.project_name,
        task_name="DataProcessing",
    )

    Task.init(
        project_name=cfg.project_name,
        task_name="BaselinesEvaluation",
        task_type=TaskTypes.custom,
        tags=cfg.tags,
        reuse_last_task_id=False,
    )

    log.info("Loading data")
    train = task_data_processing.artifacts["train"].get()
    val = task_data_processing.artifacts["validation"].get()
    log.info("Data loaded successfully!")

    log.info("Computing baselines")
    evaluate_baselines(train, val)
    log.info("Baselines computed successfully!")


if __name__ == "__main__":
    main()
