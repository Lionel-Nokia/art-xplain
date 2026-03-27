from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import streamlit as st

from src.retrieval import StyleRetriever
from src.utils import load_config, resolve_project_path, resolve_stored_path


@st.cache_resource
def get_retriever() -> StyleRetriever:
    # Le retriever recharge un modèle Keras et plusieurs artefacts NumPy.
    # Le cacher comme ressource évite un coût de réinitialisation important
    # à chaque rerun Streamlit.
    return StyleRetriever()


def coerce_object_array(values) -> np.ndarray:
    return np.asarray(values, dtype=object)


def build_style_names(labels: np.ndarray, classnames: np.ndarray) -> list[str]:
    # On sécurise ici la conversion label -> nom de style car les tableaux `.npy`
    # peuvent contenir des types hétérogènes selon la manière dont ils ont été générés.
    safe_classnames = [str(classname) for classname in np.asarray(classnames, dtype=object).tolist()]
    style_names: list[str] = []
    for label in labels:
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            style_names.append(str(label))
            continue
        if 0 <= label_int < len(safe_classnames):
            style_names.append(safe_classnames[label_int])
        else:
            style_names.append(str(label_int))
    return style_names


def find_best_index(filenames: np.ndarray, best_filepath: str) -> int | None:
    # Double stratégie de matching :
    # - égalité stricte sur le chemin complet quand les artefacts sont synchronisés,
    # - repli sur le nom de fichier quand les chemins ont été résolus différemment.
    target = str(best_filepath)
    for idx, filename in enumerate(filenames):
        if str(filename) == target or Path(str(filename)).name == Path(target).name:
            return idx
    return None


def prettify_token(text: str) -> str:
    token = str(text).strip().replace("_", " ")
    token = " ".join(token.split())
    return token if token else "Inconnu"


def extract_artist_and_title(filepath: str) -> tuple[str, str]:
    # Convention de nommage métier :
    # `<artiste>_<tableau>.jpg`.
    # Si elle n'est pas respectée, on préfère remonter "Inconnu"
    # plutôt que de produire un parsing faux mais plausible.
    filename = Path(str(filepath)).stem
    if "_" not in filename:
        return "Inconnu", prettify_token(filename)
    artist_slug, title_slug = filename.split("_", 1)
    return prettify_token(artist_slug), prettify_token(title_slug)


def format_explanation_layer_options(layer_names: list[str]) -> tuple[list[str], dict[str, str]]:
    # Les noms de couches bruts sont parlants pour les développeurs,
    # mais peu pour les utilisateurs. On leur associe donc un libellé
    # pédagogique basé sur la profondeur relative dans le réseau.
    if not layer_names:
        return [], {}
    total = len(layer_names)
    labels_by_name: dict[str, str] = {}
    for idx, layer_name in enumerate(layer_names):
        ratio = idx / max(total - 1, 1)
        if ratio <= 0.33:
            family = "Couche precoce"
        elif ratio <= 0.66:
            family = "Couche intermediaire"
        else:
            family = "Couche profonde"
        labels_by_name[layer_name] = f"{family} {idx + 1}/{total}"
    return layer_names, labels_by_name


def select_explanation_layers(layer_names: list[str], layer_numbers: list[int]) -> tuple[list[tuple[int, str]], list[int]]:
    if not layer_names:
        return [], layer_numbers
    selected_layers: list[tuple[int, str]] = []
    missing_layers: list[int] = []
    for layer_number in layer_numbers:
        idx = layer_number - 1
        if 0 <= idx < len(layer_names):
            selected_layers.append((layer_number, layer_names[idx]))
        else:
            missing_layers.append(layer_number)
    return selected_layers, missing_layers


def build_random_gradcam_layer_numbers(pair_count: int, min_layer: int = 1, max_layer: int = 245) -> list[int]:
    # On échantillonne les couches sur tout l'intervalle disponible
    # afin d'obtenir une "histoire" Grad-CAM couvrant couches précoces,
    # intermédiaires et profondes, plutôt qu'un tirage concentré sur une zone.
    count = max(1, int(pair_count))
    min_layer = max(1, int(min_layer))
    max_layer = max(min_layer, int(max_layer))
    if count == 1:
        return [random.randint(min_layer, max_layer)]

    total_span = max_layer - min_layer
    segment_size = total_span / count
    selected_numbers: list[int] = []
    for index in range(count):
        segment_start = min_layer + (segment_size * index)
        segment_end = min_layer + (segment_size * (index + 1))
        low = int(round(segment_start))
        high = int(round(segment_end))
        if index == 0:
            low = min_layer
        if index == count - 1:
            high = max_layer
        low = max(min_layer, low)
        high = min(max_layer, max(low, high))
        selected_numbers.append(random.randint(low, high))

    unique_numbers = sorted(set(selected_numbers))
    while len(unique_numbers) < count and len(unique_numbers) < (max_layer - min_layer + 1):
        candidate = random.randint(min_layer, max_layer)
        if candidate not in unique_numbers:
            unique_numbers.append(candidate)
            unique_numbers.sort()
    return unique_numbers


@st.cache_data
def load_umap_bundle(bundle_path: str) -> dict[str, np.ndarray]:
    with np.load(bundle_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


@st.cache_data
def load_numpy_array(path: str, allow_pickle: bool = False) -> np.ndarray:
    return np.load(path, allow_pickle=allow_pickle)


def load_latent_and_meta():
    # Cette fonction prépare les données nécessaires à la visualisation UMAP.
    # Elle gère deux formats d'entrée :
    # - un bundle `.npz` moderne,
    # - ou les anciens fichiers `.npy` séparés.
    # Ce choix limite la casse pendant les transitions d'artefacts dans le projet.
    cfg = load_config()
    emb_root = resolve_project_path(cfg["paths"]["embeddings_root"])
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def update_progress(step: int, total: int, message: str) -> None:
        progress_text.caption(f"Chargement des espaces latents : {message}")
        progress_bar.progress(step / total)

    bundle_path = emb_root / "umap_bundle.npz"
    try:
        if bundle_path.exists():
            update_progress(1, 5, "ouverture du bundle UMAP")
            data = load_umap_bundle(str(bundle_path))
            required_keys = {"latent_2d", "labels", "classnames", "filenames"}
            update_progress(2, 5, "vérification du contenu du bundle")
            missing = required_keys.difference(set(data.keys()))
            if missing:
                raise ValueError(f"Le bundle UMAP est incomplet. Clés manquantes: {sorted(missing)}")
            update_progress(3, 5, "lecture de la projection UMAP")
            latent_2d = np.asarray(data["latent_2d"])
            update_progress(4, 5, "lecture des labels et métadonnées")
            labels = np.asarray(data["labels"])
            classnames = coerce_object_array(data["classnames"])
            filenames = coerce_object_array([str(resolve_stored_path(fp)) for fp in data["filenames"]])
        else:
            latent_path = emb_root / "latent_2d.npy"
            if not latent_path.exists():
                progress_text.empty()
                progress_bar.empty()
                return None
            update_progress(1, 5, "lecture de la projection UMAP")
            latent_2d = load_numpy_array(str(latent_path))
            update_progress(2, 5, "lecture des labels")
            labels = load_numpy_array(str(emb_root / "labels.npy"))
            update_progress(3, 5, "lecture des noms de styles")
            classnames = load_numpy_array(str(emb_root / "classnames.npy"), allow_pickle=True)
            update_progress(4, 5, "lecture des chemins d'oeuvres")
            filenames = load_numpy_array(str(emb_root / "filenames.npy"), allow_pickle=True)
            latent_2d = np.asarray(latent_2d)
            labels = np.asarray(labels)
            classnames = coerce_object_array(classnames)
            filenames = coerce_object_array([str(resolve_stored_path(fp)) for fp in filenames])

        update_progress(5, 5, "validation finale")
        n_latent = int(latent_2d.shape[0])
        n_labels = int(len(labels))
        n_filenames = int(len(filenames))
        # Validation structurelle importante :
        # si ces tailles divergent, l'UMAP peut s'afficher avec des tooltips faux
        # ou surligner la mauvaise œuvre. On préfère donc échouer tôt ici.
        if latent_2d.ndim != 2 or latent_2d.shape[1] < 2:
            raise ValueError(f"Projection UMAP invalide: shape={latent_2d.shape}, attendu=(N, 2)")
        if not (n_latent == n_labels == n_filenames):
            raise ValueError(
                "Incohérence entre latent_2d, labels et filenames.\n"
                f" - latent_2d : {n_latent}\n - labels    : {n_labels}\n - filenames : {n_filenames}\n"
                "Relance compute_embeddings.py puis visualization_umap.py."
            )
        return latent_2d, labels, classnames, filenames
    finally:
        progress_text.empty()
        progress_bar.empty()
