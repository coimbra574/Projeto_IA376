.ONESHELL:
SHELL = /bin/bash

.PHONY: clean data lint requirements setup sample

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = ia376-diffusion
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip pip-tools setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --ignore-installed
	$(PYTHON_INTERPRETER) -m pip install -r src/submodules/denoising-diffusion-gan/requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -r src/submodules/stylegan2-pytorch/requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src
	black src

## Set up python interpreter environment
create_environment:
	conda env create -f environment.yml --force

setup: setup_project setup_ddgan setup_wgan setup_stylegan2

setup_project:
	$(PYTHON_INTERPRETER) -m pip install -e .

setup_ddgan:
	cd src/submodules/denoising-diffusion-gan/; $(PYTHON_INTERPRETER) -m pip install -e .

setup_wgan:
	cd src/submodules/WassersteinGAN/; $(PYTHON_INTERPRETER) -m pip install -e .

setup_stylegan2:
	cd src/submodules/stylegan2-pytorch/; $(PYTHON_INTERPRETER) -m pip install -e .

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RESULTS                                                               #
#################################################################################

sample: sample_original sample_ddgan sample_wgan # sample_stylegan

sample_original:
	$(PYTHON_INTERPRETER) src/data/generate_samples.py original_data --invert_p 0.3 ;
	$(PYTHON_INTERPRETER) src/data/generate_samples.py original_data --invert_p 0.5 ;
	$(PYTHON_INTERPRETER) src/data/generate_samples.py original_data --invert_p 0.7

sample_ddgan:
	$(PYTHON_INTERPRETER) src/data/generate_samples.py ddgan --weights_path models/ddgan/mnist_0.3/netG_200.pth --params_path models/ddgan/mnist_0.3/params.json ;
	$(PYTHON_INTERPRETER) src/data/generate_samples.py ddgan --weights_path models/ddgan/mnist_0.5/netG_200.pth --params_path models/ddgan/mnist_0.5/params.json ;
	$(PYTHON_INTERPRETER) src/data/generate_samples.py ddgan --weights_path models/ddgan/mnist_0.7/netG_200.pth --params_path models/ddgan/mnist_0.7/params.json

sample_stylegan: 
	$(PYTHON_INTERPRETER) src/data/generate_samples.py stylegan2 --weights_path models/stylegan/mnist_0.3/model_700.pth --params_path models/stylegan/mnist_0.3/params.json ;
	$(PYTHON_INTERPRETER) src/data/generate_samples.py stylegan2 --weights_path models/stylegan/mnist_0.5/model_700.pth --params_path models/stylegan/mnist_0.5/params.json ;
	$(PYTHON_INTERPRETER) src/data/generate_samples.py stylegan2 --weights_path models/stylegan/mnist_0.7/model_700.pth --params_path models/stylegan/mnist_0.7/params.json


sample_wgan: 
	 $(PYTHON_INTERPRETER) src/data/generate_samples.py wgan --weights_path models/wgan/mnist_0.5/model.pth --params_path models/wgan/mnist_0.5/params.json 

distributions: 
	$(PYTHON_INTERPRETER) src/data/compute_distribution.py data/generated_samples data/processed


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
