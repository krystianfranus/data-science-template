import hydra
import mlflow as mf
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="preprocess")
def main(config: DictConfig):
    with mf.start_run():
        data = pd.read_csv("data/raw_data/data.csv")
        mask = np.random.rand(len(data)) < config["split_ratio"]
        train_data = data[mask]
        test_data = data[~mask]

        train_data.to_csv("data/processed/train.csv", index=False)
        test_data.to_csv("data/processed/test.csv", index=False)

        # Log params
        params = {
            "split_ratio": config["split_ratio"],
            "train_size": len(train_data),
            "test_size": len(test_data),
        }
        mf.log_params(params)


if __name__ == "__main__":
    main()
