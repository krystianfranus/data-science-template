import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "baselines"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    baselines_method_name = cfg.method._target_.split(".")[-1]

    task_data_processing = Task.get_task(
        task_id=cfg.data_processing_task_id,
        project_name=cfg.project_name,
        task_name="DataProcessing",
    )

    Task.init(
        project_name=cfg.project_name,
        task_name="Baselines",
        task_type=TaskTypes.custom,
        tags=[baselines_method_name],
        reuse_last_task_id=False,
        output_uri=task_data_processing.output_uri,
    )

    log.info("Loading data")
    train = task_data_processing.artifacts["train"].get()
    val = task_data_processing.artifacts["val"].get()
    log.info("Data loaded successfully!")

    log.info(f"Computing baselines using '{baselines_method_name}' method")
    hydra.utils.call(cfg.method, train=train, val=val)
    log.info("Baselines computed successfully!")


if __name__ == "__main__":
    main()
