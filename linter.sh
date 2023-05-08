#!/bin/bash

echo "Running black ..."
black .

echo "Running isort ..."
isort .

echo "Running flake8 ..."
flake8 .
