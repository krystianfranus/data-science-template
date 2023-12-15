import logging

import hydra
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig

from mypackage import get_project_root
from mypackage.baseline.baseline import compute_baseline

load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(get_project_root() / "configs" / "baseline"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    Task.init(
        project_name="MyProject",
        task_name="Baseline",
        task_type=TaskTypes.custom,
        reuse_last_task_id=False,
    )

    task_prev = Task.get_task(
        task_id=cfg.prev_task_id,
        project_name="MyProject",
        task_name="Preprocessing",
    )

    log.info("Loading data")
    train = task_prev.artifacts["train"].get()
    val = task_prev.artifacts["val"].get()
    log.info("Loading data - success!")

    log.info("Computing baselines")
    compute_baseline(train, val, cfg.data_type)
    log.info("Computing baselines - success!")


if __name__ == "__main__":
    main()
