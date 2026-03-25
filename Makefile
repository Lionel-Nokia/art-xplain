# utilise toujours le Python du venv pyenv ;
# fonctionne sur machine GPU et machine CPU-only ;
# n’ajoute LD_LIBRARY_PATH que si un GPU TensorFlow est détecté ;
# évite d’utiliser le python ou streamlit système par erreur.
#
#
# make build_env
# make test_env
# make test_gpu
# make train

SHELL := $(or $(shell command -v zsh 2>/dev/null),$(shell command -v bash 2>/dev/null),/bin/sh)
.DEFAULT_GOAL := help

PYENV_ROOT := $(HOME)/.pyenv
PYTHON_VERSION := 3.12.9
ENV_NAME := art-xplain
PROJECT_DIR := art-xplain

ENV_BIN := $(PYENV_ROOT)/versions/$(ENV_NAME)/bin
ENV_PYTHON := $(ENV_BIN)/python
ENV_STREAMLIT := $(ENV_PYTHON) -m streamlit

.PHONY: help install build_env cleanall dataset train embeddings umap run build_model download_model get_wikiart test_gpu test_env

help:
        @echo "Commandes disponibles :"
        @echo "  make install        - enchaine build_env, cleanall, get_wikiart et dataset"
        @echo "  make build_env      - cree/verifie l'environnement pyenv '$(ENV_NAME)' et installe requirements.txt"
        @echo "  make cleanall       - supprime data/out, models et embeddings"
        @echo "  make dataset        - reconstruit le dataset avec nettoyage prealable"
        @echo "  make train          - entraine l'encodeur"
        @echo "  make embeddings     - calcule les embeddings"
        @echo "  make umap           - calcule la projection UMAP"
        @echo "  make download_model - telecharge et decompresse embeddings et models"
        @echo "  make get_wikiart    - telecharge WikiArt via Kaggle et reinitialise data/in"
        @echo "  make run            - lance l'application Streamlit"
        @echo "  make test_env       - verifie que le bon Python/venv est utilise"
        @echo "  make test_gpu       - verifie si TensorFlow detecte un GPU"
        @echo "  make build_model    - enchaine train, embeddings et umap"

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
        @"$(ENV_PYTHON)" -m pip install --upgrade pip setuptools wheel
        @"$(ENV_PYTHON)" -m pip install -r requirements.txt

cleanall:
        @echo "Suppression de $(PROJECT_DIR)/data/out, $(PROJECT_DIR)/models et $(PROJECT_DIR)/embeddings..."
        @rm -rf "$(PROJECT_DIR)/data/out"
        @rm -rf "$(PROJECT_DIR)/models"
        @rm -rf "$(PROJECT_DIR)/embeddings"

dataset:
        @cd "$(PROJECT_DIR)" && "$(ENV_PYTHON)" -m src.build_dataset_from_csv --clean-out

train:
        @cd "$(PROJECT_DIR)" && \
        if "$(ENV_PYTHON)" -c 'import tensorflow as tf; raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 1)' >/dev/null 2>&1; then \
                TFDIR="$$( "$(ENV_PYTHON)" -c 'import tensorflow, os; print(os.path.dirname(tensorflow.__file__))' )"; \
                export LD_LIBRARY_PATH="$$TFDIR:$$LD_LIBRARY_PATH"; \
                echo "GPU TensorFlow detecte - execution avec LD_LIBRARY_PATH configure"; \
        else \
                echo "Pas de GPU TensorFlow detecte - execution en mode CPU"; \
        fi; \
        "$(ENV_PYTHON)" -m src.build_encoder_model

embeddings:
        @cd "$(PROJECT_DIR)" && \
        if "$(ENV_PYTHON)" -c 'import tensorflow as tf; raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 1)' >/dev/null 2>&1; then \
                TFDIR="$$( "$(ENV_PYTHON)" -c 'import tensorflow, os; print(os.path.dirname(tensorflow.__file__))' )"; \
                export LD_LIBRARY_PATH="$$TFDIR:$$LD_LIBRARY_PATH"; \
                echo "GPU TensorFlow detecte - execution avec LD_LIBRARY_PATH configure"; \
        else \
                echo "Pas de GPU TensorFlow detecte - execution en mode CPU"; \
        fi; \
        "$(ENV_PYTHON)" -m src.compute_embeddings

umap:
        @cd "$(PROJECT_DIR)" && \
        if "$(ENV_PYTHON)" -c 'import tensorflow as tf; raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 1)' >/dev/null 2>&1; then \
                TFDIR="$$( "$(ENV_PYTHON)" -c 'import tensorflow, os; print(os.path.dirname(tensorflow.__file__))' )"; \
                export LD_LIBRARY_PATH="$$TFDIR:$$LD_LIBRARY_PATH"; \
                echo "GPU TensorFlow detecte - execution avec LD_LIBRARY_PATH configure"; \
        else \
                echo "Pas de GPU TensorFlow detecte - execution en mode CPU"; \
        fi; \
        "$(ENV_PYTHON)" -m src.visualization_umap

download_model:
        @if [ ! -d "/data" ]; then \
                cd "$(PROJECT_DIR)" && wget https://art-xplain.s3.eu-west-3.amazonaws.com/embeddings.zip; \
                cd "$(PROJECT_DIR)" && unzip embeddings.zip; \
                cd "$(PROJECT_DIR)" && rm embeddings.zip; \
                cd "$(PROJECT_DIR)" && wget https://art-xplain.s3.eu-west-3.amazonaws.com/models.zip; \
                cd "$(PROJECT_DIR)" && unzip models.zip; \
                cd "$(PROJECT_DIR)" && rm models.zip; \
                mkdir -p "$(PROJECT_DIR)/data/"; \
                cd "$(PROJECT_DIR)" && wget https://art-xplain.s3.eu-west-3.amazonaws.com/out.zip; \
                cd "$(PROJECT_DIR)" && unzip out.zip; \
                cd "$(PROJECT_DIR)" && rm out.zip; \
        else \
                mkdir -p "$(PROJECT_DIR)/data"; \
                rm -rf "$(PROJECT_DIR)/data/out"; \
                rm -rf "$(PROJECT_DIR)/embeddings"; \
                rm -rf "$(PROJECT_DIR)/models"; \
                ln -s /data/out "$(PROJECT_DIR)/data/out"; \
                ln -s /data/embeddings "$(PROJECT_DIR)/embeddings"; \
                ln -s /data/models "$(PROJECT_DIR)/models"; \
        fi

get_wikiart:
        @./scripts/get_wikiart "$(PROJECT_DIR)"

run:
        @cd "$(PROJECT_DIR)" && \
        if "$(ENV_PYTHON)" -c 'import tensorflow as tf; raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 1)' >/dev/null 2>&1; then \
                TFDIR="$$( "$(ENV_PYTHON)" -c 'import tensorflow, os; print(os.path.dirname(tensorflow.__file__))' )"; \
                export LD_LIBRARY_PATH="$$TFDIR:$$LD_LIBRARY_PATH"; \
                echo "GPU TensorFlow detecte - lancement Streamlit avec LD_LIBRARY_PATH configure"; \
        else \
                echo "Pas de GPU TensorFlow detecte - lancement Streamlit en mode standard"; \
        fi; \
        $(ENV_STREAMLIT) run src/app_streamlit.py

test_env:
        @echo "Python utilise :"
        @which "$(ENV_PYTHON)"
        @"$(ENV_PYTHON)" -c 'import sys; print(sys.executable)'
        @"$(ENV_PYTHON)" -c 'import sys; print(sys.version)'

test_gpu:
        @cd "$(PROJECT_DIR)" && \
        if ! "$(ENV_PYTHON)" -c 'import tensorflow' >/dev/null 2>&1; then \
                echo "TensorFlow n'est pas installe dans $(ENV_NAME)."; \
                exit 1; \
        fi; \
        if "$(ENV_PYTHON)" -c 'import tensorflow as tf; raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 1)' >/dev/null 2>&1; then \
                TFDIR="$$( "$(ENV_PYTHON)" -c 'import tensorflow, os; print(os.path.dirname(tensorflow.__file__))' )"; \
                export LD_LIBRARY_PATH="$$TFDIR:$$LD_LIBRARY_PATH"; \
        fi; \
        "$(ENV_PYTHON)" -c 'import tensorflow as tf; print("TF:", tf.__version__); print("Built with CUDA:", tf.test.is_built_with_cuda()); print("GPUs:", tf.config.list_physical_devices("GPU"))'

install: build_env cleanall get_wikiart dataset

build_model: train embeddings umap
