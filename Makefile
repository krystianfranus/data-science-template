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
ifeq (,$(shell which conda))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name ds-template python=3.10.4
else
	@echo ">>> Install conda first."
endif
	@echo ">>> New conda env created. Activate with:\nconda activate ds-template"

## Install Python Dependencies
install:
	pip install -e .

## Run pipeline
run:
	mlflow run . --experiment-name "pipeline"

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
