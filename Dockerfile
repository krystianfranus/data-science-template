FROM python:3.10
WORKDIR src
COPY . .

RUN pip install --upgrade pip
RUN pip install -e .