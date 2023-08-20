<div align="center">

# 🚀⚡🔥 data-science-template 🚀⚡🔥

</div>


[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai)
[![hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io)
[![tests](https://github.com/krystianfranus/data-science-template/actions/workflows/workflow.yaml/badge.svg)](https://github.com/krystianfranus/data-science-template/actions/workflows/workflow.yaml)


Hi there! This is recommender system project using:
- `lightning` for model training
- `hydra` for configuration
- `clearml` for ML cycle control
- (optionally) aws s3 as remote storage

## Prerequisites

To make use of this repository, you must first set up a ClearML server and, optionally, AWS S3 storage.

## Configuration and installation

Prepare environment variables in .env (see .env.example):
```
CLEARML_CONFIG_FILE=clearml.conf
CLEARML_API_ACCESS_KEY=<your-clearml-api-access-key>
CLEARML_API_SECRET_KEY=<your-clearml-api-secret-key>
...
```

Install with pip:
```bash
pip install .  # Add flag -e to install in editable mode
```

Using docker compose:
```bash
docker compose up -d  # Run container based on docker-compose.yml
```

Using plain docker:
```bash
docker build -t ds-image .  # Build image defined in Dockerfile 
docker run -dit --gpus all --name ds-container ds-image  # Run container based on that image
```

## Quick start

```bash
python steps/preprocess.py type=simple
python steps/compute_baseline.py type=simple
python steps/train.py experiment=simple_mlp
python steps/infer.py model_type=simple_mlp
```

## Other useful commands:

```bash
docker exec -it ds-container bash  # Execute bash in a running container
docker compose start/stop/down
docker builder prune  # Remove build cache
bash linter.sh
```

## References

* https://drivendata.github.io/cookiecutter-data-science
* https://github.com/khuyentran1401/data-science-template

## Trash
(Obsolete) Remember about clearml agent based on docker image:

```bash
clearml-agent daemon --queue default --docker my-docker-image
```
