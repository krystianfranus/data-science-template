import os

import hydra
import mlflow as mf
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config: DictConfig):

    with mf.start_run():
        exp_name = "my_experiment"

        # Preprocess
        params = {"split_ratio": config["split_ratio"]}
        preprocess = mf.run(".", "preprocess", parameters=params, run_name="preprocess", experiment_name=exp_name)
        preprocess = mf.tracking.MlflowClient().get_run(preprocess.run_id)
        train_path = os.path.join(preprocess.info.artifact_uri, "train.csv")
        test_path = os.path.join(preprocess.info.artifact_uri, "test.csv")

        # Train
        params = {
            "train_path": train_path,
            "test_path": test_path,
            "n_steps": config["n_steps"],
            "lr": config["lr"],
        }
        mf.run(".", "train", parameters=params, run_name="train", experiment_name=exp_name)


if __name__ == "__main__":
    main()
