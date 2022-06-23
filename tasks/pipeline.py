import os

import hydra
import mlflow as mf
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs/", config_name="pipeline")
def main(config: DictConfig):

    with mf.start_run() as active_run:
        # Preprocess
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


if __name__ == "__main__":
    main()
