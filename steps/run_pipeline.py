from clearml.automation import PipelineController
from dotenv import load_dotenv

load_dotenv()


def main():
    pipe = PipelineController(name="MyPipeline", project="MyProject", version="0.1.0")

    pipe.add_step(
        name="DataProcessing",
        base_task_project="MyProject",
        base_task_name="DataProcessing",
        cache_executed_step=True,
    )

    pipe.add_step(
        name="BaselinesEvaluation",
        base_task_project="MyProject",
        base_task_name="BaselinesEvaluation",
        cache_executed_step=True,
        parents=["DataProcessing"],
        parameter_override={
            "Hydra/prev_task_id": "${DataProcessing.id}"
        },  # It is preferable
        # parameter_override={
        #     "Args/overrides": "['prev_task_id=${DataProcessing.id}']"
        # },  # It also works
    )

    pipe.add_step(
        name="Training",
        base_task_project="MyProject",
        base_task_name="Training",
        cache_executed_step=True,
        parents=["DataProcessing"],
        parameter_override={"Hydra/prev_task_id": "${DataProcessing.id}"},
    )

    # pipe.add_step(
    #     name="Inference",
    #     base_task_project="MyProject",
    #     base_task_name="Inference",
    #     cache_executed_step=True,
    #     parents=["Training"],
    #     parameter_override={"Hydra/prev_task_id": "${Training.id}"},
    # )

    # for debugging purposes use local jobs
    # pipe.start_locally(run_pipeline_steps_locally=False)
    pipe.start_locally(run_pipeline_steps_locally=True)

    # Starting the pipeline (in the background) with the agent
    # pipe.set_default_execution_queue("default")
    # pipe.start("default")


if __name__ == "__main__":
    main()
