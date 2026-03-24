SHELL := /bin/zsh
.DEFAULT_GOAL := help

PYENV_ROOT := $(HOME)/.pyenv
PYTHON_VERSION := 3.12.9
ENV_NAME := art-xplain
PROJECT_DIR := art-xplain
ENV_PYTHON := $(PYENV_ROOT)/versions/$(ENV_NAME)/bin/python

.PHONY: help install build_env cleanall dataset train embeddings umap run build_model fix_model get_wikiart

help:
	@echo "Commandes disponibles :"
	@echo "  make install      - enchaine build_env, cleanall, get_wikiart et dataset"
	@echo "  make build_env    - cree/verifie l'environnement pyenv 'art-xplain' et installe requirements.txt"
	@echo "  make cleanall     - supprime data/out, models et embeddings"
	@echo "  make dataset      - reconstruit le dataset avec nettoyage prealable"
	@echo "  make train        - entraine l'encodeur"
	@echo "  make embeddings   - calcule les embeddings"
	@echo "  make umap         - calcule la projection UMAP"
	@echo "  make fix_model    - telecharge et decompresse embeddings et models"
	@echo "  make get_wikiart  - telecharge WikiArt via Kaggle et reinitialise data/in"
	@echo "  make run          - lance l'application Streamlit"
	@echo "  make build_model  - enchaine train, embeddings et umap"

build_env:
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "pyenv est requis mais introuvable."; \
		exit 1; \
	fi
	@if ! pyenv versions --bare | grep -Fxq "$(PYTHON_VERSION)"; then \
		echo "Installation de Python $(PYTHON_VERSION) via pyenv..."; \
		pyenv install "$(PYTHON_VERSION)"; \
	fi
	@if ! pyenv virtualenvs --bare | grep -Fxq "$(ENV_NAME)"; then \
		echo "Creation de l'environnement $(ENV_NAME)..."; \
		pyenv virtualenv "$(PYTHON_VERSION)" "$(ENV_NAME)"; \
	else \
		echo "Environnement $(ENV_NAME) deja present."; \
	fi
	@echo "Installation des dependances dans $(ENV_NAME)..."
	@"$(ENV_PYTHON)" -m pip install --upgrade pip
	@"$(ENV_PYTHON)" -m pip install -r requirements.txt

cleanall:
	@echo "Suppression de $(PROJECT_DIR)/data/out, $(PROJECT_DIR)/models et $(PROJECT_DIR)/embeddings..."
	@rm -rf "$(PROJECT_DIR)/data/out"
	@rm -rf "$(PROJECT_DIR)/models"
	@rm -rf "$(PROJECT_DIR)/embeddings"

dataset:
	@cd "$(PROJECT_DIR)" && python -m src.build_dataset_from_csv --clean-out

train:
	@cd "$(PROJECT_DIR)" && python -m src.build_encoder_model

embeddings:
	@cd "$(PROJECT_DIR)" && python -m src.compute_embeddings

umap:
	@cd "$(PROJECT_DIR)" && python -m src.visualization_umap

fix_model:
	@cd "$(PROJECT_DIR)" && wget https://art-xplain.s3.eu-west-3.amazonaws.com/embeddings.zip
	@cd "$(PROJECT_DIR)" && unzip embeddings.zip
	@cd "$(PROJECT_DIR)" && rm embeddings.zip
	@cd "$(PROJECT_DIR)" && wget https://art-xplain.s3.eu-west-3.amazonaws.com/models.zip
	@cd "$(PROJECT_DIR)" && unzip models.zip
	@cd "$(PROJECT_DIR)" && rm models.zip

get_wikiart:
	@./scripts/get_wikiart "$(PROJECT_DIR)"

run:
	@cd "$(PROJECT_DIR)" && streamlit run src/app_streamlit.py

install: build_env cleanall get_wikiart dataset

build_model: train embeddings umap
