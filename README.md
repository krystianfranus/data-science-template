# data-science-template

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

## Warning

Remember that ec2 ip changes every time the machine is being restarted!

So you need to apply changes in /home/krystian/clearml.conf (both host address and credentials)

Direct browser to its web server URL: http://<Server Address>:8080

More info: https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_aws_ec2_ami

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