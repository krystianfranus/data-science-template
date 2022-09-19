# data-science-template

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

## Installation

```shell
make create_env
conda activate ds-template

make install
```

## Quick start

```shell
make preprocess
make train
make predict
make pipeline
```

Remember about clearml agent based on docker image:

```shell
clearml-agent daemon --queue default --docker my-docker-image
```

Building docker image:

```shell
docker build --tag my-docker-image .
```




## References

* https://drivendata.github.io/cookiecutter-data-science
* https://github.com/khuyentran1401/data-science-template