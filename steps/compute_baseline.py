import logging
import os

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.baseline.baseline import compute_baseline

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(get_project_root(), "configs", "baseline"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    Task.init(
        project_name="MyProject",
        task_name="Baseline",
        task_type=TaskTypes.custom,
        reuse_last_task_id=False,
        output_uri=None,
    )

    task_prev = Task.get_task(project_name="MyProject", task_name="Preprocessing")
    if cfg.prev_task_id:
        task_prev = Task.get_task(task_id=cfg.prev_task_id)

    log.info("Loading data")
    train = task_prev.artifacts["train"].get()
    val = task_prev.artifacts["val"].get()

    log.info("Computing baselines")
    compute_baseline(train, val, cfg.type)

    log.info("Done!")


if __name__ == "__main__":
    main()
