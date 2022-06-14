.PHONY: create_env install lint clean

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
	conda create --name ds-template python=3.8.12
else
	@echo ">>> Install conda first."
endif
	@echo ">>> New conda env created. Activate with:\nconda activate ds-template"

## Install Python Dependencies
install:
	pip install -e .

## Lint using flake8
lint:
	flake8 src

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## TODO: pytest