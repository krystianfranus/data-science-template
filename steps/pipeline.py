from clearml.automation import PipelineController


def main():
    pipe = PipelineController(
        name="My Pipeline", project="ds_template", version="0.0.1"
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="preprocessing_pipe",
        base_task_project="ds_template",
        base_task_name="preprocessing",
        cache_executed_step=True,
    )

    pipe.add_step(
        name="training_pipe",
        base_task_project="ds_template",
        base_task_name="training",
        cache_executed_step=True,
        parents=["preprocessing_pipe"],
        parameter_override={
            "Args/overrides": "['prev_task_id=${preprocessing_pipe.id}', 'datamodule.num_workers=5', 'datamodule.pin_memory=true']"  # noqa
        },
    )

    pipe.add_step(
        name="prediction_pipe",
        base_task_project="ds_template",
        base_task_name="prediction",
        cache_executed_step=True,
        parents=["training_pipe"],
        parameter_override={
            "Args/overrides": "['prev_task_id=${training_pipe.id}', 'num_workers=5', 'pin_memory=true']"  # noqa
        },
    )

    # for debugging purposes use local jobs
    pipe.start_locally()

    # Starting the pipeline (in the background)
    # pipe.start("default")


if __name__ == "__main__":
    main()
