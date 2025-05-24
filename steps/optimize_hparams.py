from clearml import Task
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from dotenv import load_dotenv

load_dotenv()


def main():
    task_training = Task.get_task(project_name="MyProject", task_name="Training")

    Task.init(
        project_name="MyProjectHPO",
        task_name="Automatic Hyper-Parameter Optimization",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=True,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=task_training.id,
        hyper_parameters=[
            # DiscreteParameterRange("Hydra/model.n_factors", values=[8, 16]),
            # DiscreteParameterRange("Hydra/model.n_layers", values=[3, 4]),
            DiscreteParameterRange("Hydra/model.dropout", values=[0.0, 0.5]),
            DiscreteParameterRange(
                "Hydra/model.lr1", values=[0.000005, 0.00001, 0.00005, 0.0001]
            ),
            DiscreteParameterRange(
                "Hydra/model.lr2", values=[0.000005, 0.00001, 0.00005, 0.0001]
            ),
            DiscreteParameterRange(
                "Hydra/model.weight_decay", values=[0.01, 0.05, 0.1]
            ),
            DiscreteParameterRange("Hydra/trainer.max_epochs", values=[2]),
        ],
        objective_metric_title="auroc",
        objective_metric_series="val",
        objective_metric_sign="max_global",
        optimizer_class=OptimizerOptuna,
        max_number_of_concurrent_tasks=1,
        save_top_k_tasks_only=-1,
        total_max_jobs=10,
        min_iteration_per_job=2 * 857,
        max_iteration_per_job=2 * 857,
        # execution_queue="default",
        spawn_project="MyProjectHPO",
    )

    # optimizer.start(job_complete_callback=job_complete_callback)
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()


if __name__ == "__main__":
    main()
