from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    GridSearch,
    HyperParameterOptimizer,
)

task = Task.init(
    project_name="MyProjectHPO",
    task_name="Automatic Hyper-Parameter Optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=True,
)


optimizer = HyperParameterOptimizer(
    base_task_id=Task.get_task(project_name="MyProject", task_name="Training").id,
    hyper_parameters=[
        DiscreteParameterRange("Hydra/net.n_factors", values=[8, 16]),
        DiscreteParameterRange("Hydra/net.n_layers", values=[3, 4]),
        DiscreteParameterRange("Hydra/net.dropout", values=[0.0, 0.5]),
        DiscreteParameterRange("Hydra/model.lr", values=[1e-4, 1e-3]),
        DiscreteParameterRange("Hydra/model.weight_decay", values=[1e-3, 1e-2, 0]),
        DiscreteParameterRange("Hydra/trainer.max_epochs", values=[50]),
    ],
    objective_metric_title="val",
    objective_metric_series="ndcg",
    objective_metric_sign="max_global",
    optimizer_class=GridSearch,
    max_number_of_concurrent_tasks=1,
    # execution_queue="default",
    spawn_project="MyProjectHPO",
)

# optimizer.start(job_complete_callback=job_complete_callback)
optimizer.start_locally()
optimizer.wait()
optimizer.stop()
