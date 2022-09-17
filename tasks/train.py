import logging

import hydra
from omegaconf import DictConfig
from clearml import Task
from src.training.model import LinearRegression

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="train")
def main(config: DictConfig):
    task = Task.init(project_name="ds_template", task_name="training")
    task2 = Task.get_task(task_id="ee986cf15b104262a25dafe5ad30537a")

    # Load train data
    train_data = task2.artifacts["train_data"].get()
    x_train = train_data["x"]
    y_train = train_data["y"]

    # Load train data
    test_data = task2.artifacts["test_data"].get()
    x_test = test_data["x"]
    y_test = test_data["y"]

    # Train model
    n_steps = config.n_steps
    lr = config.lr
    model = LinearRegression(n_steps, lr)
    model.fit(x_train, y_train)


if __name__ == "__main__":
    main()
