.PHONY: create_env install run test clean

ifeq (,$(shell which conda))
    HAS_CONDA=False
else
    HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name ds-template python=3.10.10
	@echo ">>> New conda env created. Activate with:\nconda activate ds-template"
else
	@echo ">>> Install conda first."
endif

## Install Python Dependencies
install:
	pip install -e .

## Run preprocessing
preprocess:
	python steps/preprocess.py

## Run training
train:
	python steps/train.py experiment=implicit

## Run Inferring
infer:
	python steps/infer.py num_workers=5 pin_memory=true

## Run pipeline
pipeline:
	python steps/pipeline.py

## Run tests with pytest
test:
	pytest --cov=src tests/

## Delete all compiled Python files and logs
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf logs/
	rm -rf mlruns/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf notebooks/.ipynb_checkpoints/
