import torch
from clearml import Task
from flask import Flask, jsonify, request

from mypackage.training.models.task import SimpleMLPTask

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    task_prev = Task.get_task(project_name="MyProject", task_name="Training")
    ckpt_path = task_prev.models["input"][-1].get_local_copy()
    model = SimpleMLPTask.load_from_checkpoint(ckpt_path, map_location="cpu")

    input_data = torch.tensor(request.json)
    users, items = input_data[:, 0], input_data[:, 1]

    model.eval()
    with torch.no_grad():
        prediction = model.predict(users, items)

    response = {"prediction": prediction.tolist()}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
