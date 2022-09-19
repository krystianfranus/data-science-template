FROM python:3.10
WORKDIR src
COPY . .
RUN pip install -e .