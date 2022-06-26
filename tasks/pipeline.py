import logging
import os

import hydra
import mlflow as mf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="pipeline")
def main(config: DictConfig):
    with mf.start_run() as active_run:
        hconfig = HydraConfig.get()

        # Preprocess
        log.info("Step 1 of pipeline - Preprocess")
        params = {"split_ratio": config.split_ratio}
        preprocess = mf.run(
            ".",
            "preprocess",
            parameters=params,
            run_id=active_run.info.run_id,
        )
        preprocess = mf.tracking.MlflowClient().get_run(preprocess.run_id)
        train_data_path = os.path.join(preprocess.info.artifact_uri, "data/train.csv")
        test_data_path = os.path.join(preprocess.info.artifact_uri, "data/test.csv")

        # Train
        log.info("Step 2 of pipeline  - Train")
        params = {
            "train_data_path": train_data_path,
            "test_data_path": test_data_path,
            "n_steps": config.n_steps,
            "lr": config.lr,
        }
        mf.run(
            ".",
            "train",
            parameters=params,
            run_id=active_run.info.run_id,
        )

        # Log artifacts
        log.info(f"Logging (mlflow) artifacts into '{active_run.info.artifact_uri}'")
        mf.log_artifacts(hconfig.runtime.output_dir, artifact_path="logs/pipeline")


if __name__ == "__main__":
    main()
