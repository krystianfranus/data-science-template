from clearml import Task
from clearml.automation import PipelineController


pipe = PipelineController(name="Pipeline demo", project="ds_template", version="0.0.1")

# pipe.add_parameter(
#     "url",
#     "https://files.community.clear.ml/examples%252F.pipelines%252FPipeline%20demo/stage_data.8f17b6316ce442ce8904f6fccb1763de/artifacts/dataset/f6d08388e9bc44c86cab497ad31403c4.iris_dataset.pkl",
#     "dataset_url",
# )

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="preprocessing_pipe",
    base_task_project="ds_template",
    base_task_name="preprocessing",
    # parameter_override={"General/dataset_url": "${pipeline.url}"},
)

# pipe.add_step(
#     name="stage_process",
#     parents=["stage_data"],
#     base_task_project="examples",
#     base_task_name="Pipeline step 2 process dataset",
#     parameter_override={
#         "General/dataset_url": "${stage_data.artifacts.dataset.url}",
#         "General/test_size": 0.25,
#     },
#     pre_execute_callback=pre_execute_callback_example,
#     post_execute_callback=post_execute_callback_example,
# )
# pipe.add_step(
#     name="stage_train",
#     parents=["stage_process"],
#     base_task_project="examples",
#     base_task_name="Pipeline step 3 train model",
#     parameter_override={"General/dataset_task_id": "${stage_process.id}"},
# )
#
# for debugging purposes use local jobs
pipe.start_locally()

# Starting the pipeline (in the background)
pipe.start()

print("done")
