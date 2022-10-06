from clearml.automation import PipelineController


def main():
    pipe = PipelineController(
        name="My Pipeline", project="My project", version="0.0.1"
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="Preprocessing",
        base_task_project="My project",
        base_task_name="Preprocessing",
        cache_executed_step=True,
    )

    pipe.add_step(
        name="Training",
        base_task_project="My project",
        base_task_name="Training",
        cache_executed_step=True,
        parents=["Preprocessing"],
        parameter_override={
            "Args/overrides": "['prev_task_id=${Preprocessing.id}', 'datamodule.num_workers=5', 'datamodule.pin_memory=true']"  # noqa
        },
    )

    pipe.add_step(
        name="Inference",
        base_task_project="My project",
        base_task_name="Inference",
        cache_executed_step=True,
        parents=["Training"],
        parameter_override={
            "Args/overrides": "['prev_task_id=${Training.id}', 'num_workers=5', 'pin_memory=true']"  # noqa
        },
    )

    # for debugging purposes use local jobs
    # pipe.start_locally(run_pipeline_steps_locally=False)
    pipe.start_locally(run_pipeline_steps_locally=True)

    # Starting the pipeline (in the background)
    # pipe.start("default")


if __name__ == "__main__":
    main()
