# data-science-template

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>


## Installation

Export environment variables from .env:
```
export $(cat .env | xargs)
```

Using docker compose (recommended):
```shell
docker compose up -d  # Run container based on docker-compose.yml
```

Using docker:
```shell
docker build -t ds-image .  # Build image defined in Dockerfile 
docker run -dit --gpus all --name ds-container ds-image  # Run container based on that image
```

Alternatively using pip:
```shell
pip install .  # Add flag -e to install in editable mode
```


## Quick start

```shell
python steps/preprocess.py type=simple
python steps/compute_baseline.py type=simple
python steps/train.py experiment=simple_mlp
python steps/infer.py model_type=simple_mlp
```


## ClearML (MLOps)

ClearML is used here to handle mlops tasks. Relevant environment variables should be stored in **.env** file to make it work:
```
CLEARML_CONFIG_FILE=<path_to_clearml.conf>
CLEARML_API_ACCESS_KEY=<your_access_key>
CLEARML_API_SECRET_KEY=<your_secret_key>
```


## Other useful commands:

```bash
docker exec -it ds-container bash  # Execute bash in a running container
docker compose start/stop/down
docker builder prune  # Remove build cache
```


## References

* https://drivendata.github.io/cookiecutter-data-science
* https://github.com/khuyentran1401/data-science-template


## Trash
(Obsolete) Remember about clearml agent based on docker image:

```shell
clearml-agent daemon --queue default --docker my-docker-image
```
