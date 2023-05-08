import logging
import os

import hydra
from clearml import Task, TaskTypes
from omegaconf import DictConfig

from mypackage.baseline.baseline import compute_baseline

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs", "baseline"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    Task.init(
        project_name="MyProject",
        task_name="Baseline",
        task_type=TaskTypes.custom,
        reuse_last_task_id=False,
        output_uri=None,
    )

    if cfg.prev_task_id is not None:
        task_prev = Task.get_task(task_id=cfg.prev_task_id)
    else:
        task_prev = Task.get_task(project_name="MyProject", task_name="Preprocessing")

    train = task_prev.artifacts["train"].get()
    val = task_prev.artifacts["val"].get()

    compute_baseline(train, val, cfg.type)
    log.info("Done!")


if __name__ == "__main__":
    main()
