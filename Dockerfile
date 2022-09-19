FROM python:3.10
WORKDIR /app

COPY . .
RUN pip3 install -e .