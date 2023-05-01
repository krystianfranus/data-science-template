FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages and Python 3.10
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev python3.10-venv vim git

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

## Create a virtual environment and activate it
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /project
COPY . .
# RUN chown -R root /project
RUN pip install --upgrade pip && pip install -e .
RUN jupyter contrib nbextension install
