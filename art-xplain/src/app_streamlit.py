from __future__ import annotations
# Active l'évaluation différée des annotations de type.
# Concrètement, cela permet d'écrire des annotations modernes comme list[str]
# sans que Python essaie de les résoudre immédiatement au moment du chargement.
# C'est utile pour :
# 1) améliorer la compatibilité selon les versions de Python,
# 2) éviter certains problèmes de références circulaires dans les types,
# 3) garder un code plus lisible avec les hints modernes.

import sys
# Module standard donnant accès à des informations et réglages liés à l'interpréteur Python.
# Ici, il sert surtout à manipuler sys.path, c'est-à-dire la liste des dossiers
# dans lesquels Python cherche les modules à importer.

import tempfile
# Module standard permettant de créer des fichiers temporaires.
# Dans cette application, on s'en sert pour écrire l'image uploadée par l'utilisateur
# sur le disque afin que le moteur de recherche puisse la lire comme un vrai fichier.

from pathlib import Path
# Path offre une manière moderne, robuste et lisible de manipuler des chemins de fichiers.
# C'est préférable aux simples chaînes de caractères, car on peut faire des opérations
# comme .stem, .suffix, / pour concaténer des chemins, etc.

import numpy as np
# NumPy est utilisé ici pour manipuler les tableaux numériques,
# en particulier les embeddings, labels, projections UMAP et autres structures
# de données liées au moteur de similarité.

import pandas as pd
# Pandas sert à construire des DataFrames afin d'afficher des tableaux clairs dans Streamlit,
# par exemple le résumé des résultats ou les données préparées pour la visualisation UMAP.

import plotly.express as px
# Interface haut niveau de Plotly, pratique pour créer rapidement des graphiques,
# ici notamment le nuage de points UMAP.

import plotly.graph_objects as go
# Interface plus bas niveau de Plotly, utile quand on veut ajouter des traces personnalisées,
# par exemple pour surligner le meilleur résultat (top-1) dans l'UMAP.

import streamlit as st
# Streamlit est le framework web utilisé pour créer l'application interactive.
# Toutes les instructions d'interface passent par l'objet st :
# st.title, st.image, st.dataframe, st.checkbox, st.file_uploader, etc.


# =============================================================================
# GESTION DES IMPORTS INTERNES AU PROJET
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# __file__ représente le chemin du fichier Python courant.
# Path(__file__) le convertit en objet Path.
# .resolve() transforme ce chemin en chemin absolu canonique.
# .parents[1] remonte de deux niveaux dans l'arborescence.
# L'objectif est de retrouver la racine du projet pour pouvoir importer les modules internes.
# Exemple typique : si ce script est dans project/app/app_streamlit.py,
# alors parents[1] peut pointer vers project/.

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Python cherche les imports dans les dossiers listés dans sys.path.
# On ajoute ici la racine du projet au début de cette liste (index 0)
# pour s'assurer que les imports internes comme src.utils et src.retrieval fonctionnent.
# Le test préalable évite d'ajouter plusieurs fois le même chemin.

from src.utils import load_config, resolve_project_path, resolve_stored_path
# Importe une fonction utilitaire interne censée charger la configuration du projet.
# On l'utilise plus loin pour récupérer les chemins vers les embeddings et autres ressources.

from src.retrieval import StyleRetriever
# Importe la classe centrale du moteur de recherche de similarité stylistique.
# C'est elle qui va probablement encapsuler :
# - le chargement du modèle,
# - l'extraction d'embeddings,
# - la comparaison de similarité,
# - et éventuellement Grad-CAM.


# =============================================================================
# CONFIGURATION GÉNÉRALE DE L'INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="Art-Xplain", layout="wide")
# Configure la page Streamlit avant tout rendu visuel important.
# page_title : titre de l'onglet du navigateur.
# layout="wide" : utilise toute la largeur disponible, ce qui est très utile
# pour afficher plusieurs images côte à côte et une visualisation UMAP large.

st.title("Art-Xplain — Similarité stylistique")
# Affiche le titre principal de l'application en haut de la page.
# C'est le premier élément visible par l'utilisateur.


# =============================================================================
# CHARGEMENT DU MOTEUR DE RETRIEVAL
# =============================================================================

@st.cache_resource
def get_retriever() -> StyleRetriever:
    """
    Crée et retourne l'objet StyleRetriever.

    L'utilisation de @st.cache_resource permet d'éviter de recharger
    inutilement le moteur de recherche à chaque interaction Streamlit.
    """
    return StyleRetriever()
# Cette fonction instancie le moteur principal de recherche.
# Le décorateur @st.cache_resource indique à Streamlit que le résultat
# est une ressource lourde et durable (par ex. modèle ML, index en mémoire,
# objets coûteux à construire).
# Sans ce cache, chaque interaction utilisateur (cocher une case, filtrer un style,
# uploader une image) pourrait recharger le moteur, ce qui serait très lent.
# Le type de retour annoncé -> StyleRetriever aide la lisibilité et l'autocomplétion.


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def _coerce_object_array(values) -> np.ndarray:
    """
    Convertit une collection en tableau NumPy de type object.
    """
    return np.asarray(values, dtype=object)
# np.asarray convertit 'values' en tableau NumPy sans recopier inutilement les données
# si elles sont déjà sous une forme compatible.
# dtype=object force un type très souple, utile quand les éléments ne sont pas
# purement numériques ou homogènes (par ex. chemins, noms de classes, objets divers).
# Le préfixe _ dans le nom suggère qu'il s'agit d'une fonction utilitaire interne au module.


def _build_style_names(labels: np.ndarray, classnames: np.ndarray) -> list[str]:
    """
    Transforme des labels numériques en noms de styles lisibles.
    """
    style_names: list[str] = []
    # On prépare une liste vide qui contiendra les noms de styles correspondant
    # à chaque label numérique.
    # Exemple : 0 -> "Impressionnisme", 1 -> "Cubisme", etc.

    for lab in labels:
        # On parcourt chaque label issu du jeu de données ou des embeddings.
        # labels est supposé contenir des identifiants de classe.
        try:
            idx = int(lab)
            # On essaie de convertir le label en entier.
            # C'est utile si lab arrive sous forme np.int64, float, ou même string numérique.

            if 0 <= idx < len(classnames):
                style_names.append(str(classnames[idx]))
                # Cas nominal : le label est valide et correspond bien à un index de classnames.
                # On convertit le nom de classe en string pour homogénéiser les types.
            else:
                style_names.append(f"Classe {lab}")
                # Cas de secours : le label est en dehors de la plage attendue.
                # On ne plante pas l'application ; on génère un nom générique.
        except Exception:
            style_names.append(f"Classe {lab}")
            # Si la conversion en entier échoue pour une raison quelconque,
            # on adopte aussi un nom générique.
            # L'idée générale de cette fonction est la robustesse :
            # l'interface ne doit pas casser parce qu'un label est mal formé.

    return style_names
    # Renvoie la liste finale des noms lisibles.


def _find_best_index(filenames: np.ndarray, best_filepath: str) -> int | None:
    """
    Retrouve l'indice du meilleur résultat dans la liste des fichiers UMAP.
    """
    filenames_str = np.asarray(
        [str(resolve_stored_path(fp)) for fp in filenames],
        dtype=object,
    )
    # On convertit tous les éléments de filenames en chaînes de caractères.
    # Pourquoi ? Parce qu'ils peuvent être de type Path, numpy scalar, ou autre,
    # et qu'on veut faire une comparaison homogène avec best_filepath.

    matches = np.where(filenames_str == str(best_filepath))[0]
    # np.where retourne les indices où la condition est vraie.
    # Ici, on cherche le ou les fichiers dont le chemin correspond exactement
    # au chemin du meilleur résultat retourné par le moteur.
    # Le [0] récupère le tableau des indices.

    if len(matches) == 0:
        return None
        # Aucun fichier correspondant n'a été trouvé dans la liste UMAP.
        # On renvoie None pour indiquer l'absence de correspondance.

    return int(matches[0])
    # Si plusieurs matches existent, on prend le premier.
    # En pratique, il devrait idéalement n'y en avoir qu'un seul.


def _prettify_token(text: str) -> str:
    """
    Transforme une chaîne de type slug en texte lisible.

    Exemples :
        'vincent-van-gogh' -> 'Vincent Van Gogh'
        'la-nuit-etoilee'  -> 'La Nuit Etoilee'
    """
    text = str(text).strip().replace("-", " ")
    # str(text) garantit qu'on travaille sur une chaîne.
    # .strip() retire les espaces en début et fin.
    # .replace("-", " ") remplace les tirets par des espaces
    # afin de convertir un slug en texte plus naturel.

    return text.title()
    # .title() met une majuscule à chaque mot.
    # Exemple : "vincent van gogh" -> "Vincent Van Gogh".


def _extract_artist_and_title(filepath: str) -> tuple[str, str]:
    """
    Extrait le nom de l'artiste et le nom du tableau à partir du nom de fichier.

    Format attendu :
        nom-de-l-artiste_nom-du-tableau.jpg
    """
    filename = Path(str(filepath)).stem
    # Convertit le chemin reçu en objet Path, puis récupère uniquement le nom du fichier
    # sans son extension.
    # Exemple : "/data/img/van-gogh_starry-night.jpg" -> "van-gogh_starry-night"

    if "_" not in filename:
        return "Inconnu", "Inconnu"
        # Si le séparateur '_' n'est pas présent, on ne peut pas distinguer artiste et tableau.
        # On adopte donc une stratégie de repli :
        # - artiste = "Inconnu"
        # - titre = "Inconnu"

    artist_raw, title_raw = filename.split("_", 1)
    # On découpe au premier underscore uniquement.
    # Le paramètre 1 est important : s'il y a d'autres underscores dans le titre,
    # ils restent dans la seconde partie au lieu de provoquer trop de segments.

    artist = _prettify_token(artist_raw) if artist_raw.strip() else "Inconnu"
    # Rend le nom d'artiste lisible.

    title = _prettify_token(title_raw) if title_raw.strip() else "Inconnu"
    # Rend le titre lisible.

    if artist == "Inconnu" or title == "Inconnu":
        return "Inconnu", "Inconnu"

    return artist, title


def _format_explanation_layer_options(layer_names: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Construit des libellés lisibles pour les couches Grad-CAM++.
    """
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


def _select_explanation_layers(layer_names: list[str], layer_numbers: list[int]) -> tuple[list[tuple[int, str]], list[int]]:
    """
    Selectionne des couches Grad-CAM++ par position humaine (1-based).
    """
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


# =============================================================================
# CHARGEMENT DES DONNÉES LATENTES (UMAP + MÉTADONNÉES)
# =============================================================================

@st.cache_data
def load_latent_and_meta():
    """
    Charge les données nécessaires à la visualisation UMAP interactive.

    Cette fonction centralise toute la logique de lecture des artefacts
    produits en amont par le pipeline ML.
    Elle accepte implicitement deux formats de stockage :
    1. un bundle unique `umap_bundle.npz`, pratique pour le déploiement,
    2. plusieurs fichiers `.npy`, pratique pendant le développement.

    Le décorateur `@st.cache_data` est bien adapté ici car :
    - on manipule des données sérialisées relativement stables,
    - leur chargement peut coûter du temps,
    - mais contrairement à un modèle ML vivant en mémoire, il s'agit bien
      de "données" et non d'une "ressource" applicative.
    """
    cfg = load_config()
    # Charge la configuration du projet depuis une source interne.
    # On suppose que cfg est un dictionnaire contenant notamment des chemins.

    emb_root = resolve_project_path(cfg["paths"]["embeddings_root"])
    # Récupère le dossier racine où sont stockés les embeddings et métadonnées.
    # On convertit ce chemin en Path pour manipulations propres ensuite.

    bundle_path = emb_root / "umap_bundle.npz"
    # Construit le chemin vers un fichier compressé .npz contenant potentiellement
    # toutes les données nécessaires à l'affichage UMAP dans un seul bundle.

    if bundle_path.exists():
        data = np.load(bundle_path, allow_pickle=True)
        # Si le bundle existe, on le charge.
        # np.load(..., allow_pickle=True) permet de lire des objets Python sérialisés,
        # ce qui peut être nécessaire pour classnames ou filenames.

        required_keys = {"latent_2d", "labels", "classnames", "filenames"}
        # Ensemble des clés indispensables pour que l'UMAP fonctionne correctement.

        missing = required_keys.difference(set(data.files))
        # data.files contient les clés présentes dans le fichier .npz.
        # On calcule celles qui manquent.

        if missing:
            raise ValueError(
                f"Le bundle UMAP est incomplet. Clés manquantes: {sorted(missing)}"
            )
            # On arrête explicitement avec un message clair plutôt que de laisser
            # l'application échouer plus loin de manière obscure.

        latent_2d = np.asarray(data["latent_2d"])
        # Projection 2D des embeddings, typiquement shape (N, 2).

        labels = np.asarray(data["labels"])
        # Labels numériques associés à chaque point.

        classnames = _coerce_object_array(data["classnames"])
        # Noms des classes / styles, convertis proprement en array object.

        filenames = _coerce_object_array(
            [str(resolve_stored_path(fp)) for fp in data["filenames"]]
        )
        # Chemins ou noms des fichiers correspondant à chaque point projeté.

    else:
        latent_path = emb_root / "latent_2d.npy"
        # Si le bundle complet n'existe pas, on tente un chargement à partir de fichiers séparés.

        if not latent_path.exists():
            return None
            # Si même la projection principale n'existe pas, on considère que l'UMAP
            # n'est pas disponible et on renvoie None.

        latent_2d = np.load(latent_path)
        # Charge la projection 2D.

        labels = np.load(emb_root / "labels.npy")
        # Charge les labels.

        classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)
        # Charge les noms de classes.

        filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)
        # Charge les chemins de fichiers.

        latent_2d = np.asarray(latent_2d)
        labels = np.asarray(labels)
        classnames = _coerce_object_array(classnames)
        filenames = _coerce_object_array([str(resolve_stored_path(fp)) for fp in filenames])
        # Uniformisation des types pour la suite du traitement.

    if latent_2d.ndim != 2 or latent_2d.shape[1] < 2:
        raise ValueError(
            f"Projection UMAP invalide: shape={latent_2d.shape}, attendu=(N, 2)"
        )
        # Vérifie que la projection a bien 2 dimensions au sens matrice
        # et au moins 2 colonnes.
        # UMAP destiné à un scatter 2D doit fournir un tableau de forme (N, 2).

    n_latent = int(latent_2d.shape[0])
    # Nombre de points dans la projection.

    n_labels = int(len(labels))
    # Nombre de labels.

    n_filenames = int(len(filenames))
    # Nombre de fichiers.

    if not (n_latent == n_labels == n_filenames):
        raise ValueError(
            "Incohérence entre latent_2d, labels et filenames.\n"
            f" - latent_2d : {n_latent}\n"
            f" - labels    : {n_labels}\n"
            f" - filenames : {n_filenames}\n"
            "Relance compute_embeddings.py puis visualization_umap.py."
        )
        # Contrôle de cohérence fondamental : chaque point projeté doit correspondre
        # à un label et à un fichier, ni plus ni moins.
        # Le message d'erreur guide aussi vers les scripts à relancer.

    return latent_2d, labels, classnames, filenames
    # Renvoie l'ensemble des données nécessaires à la visualisation interactive.


@st.cache_data
def _inspect_runtime_assets() -> dict[str, object]:
    """
    Prépare un résumé lisible de l'état des artefacts utilisés par l'application.
    """
    cfg = load_config()

    data_root = resolve_project_path(cfg["paths"]["keras_root"])
    emb_root = resolve_project_path(cfg["paths"]["embeddings_root"])
    models_root = resolve_project_path(cfg["paths"]["models_root"])

    def status_row(label: str, ok: bool, detail: str, state: str | None = None) -> dict[str, str]:
        return {
            "label": label,
            "state": state or ("Valide" if ok else "Indisponible"),
            "tone": "#e8f7ee" if ok else "#fdecec",
            "accent": "#1f7a45" if ok else "#b42318",
            "detail": detail,
        }

    model_path = models_root / "encoder.keras"
    model_ok = model_path.exists() and model_path.is_file()
    model_detail = (
        f"encodeur trouvé: {model_path.name}"
        if model_ok
        else f"fichier manquant: {model_path.name}"
    )

    required_embedding_files = [
        emb_root / "vectors.npy",
        emb_root / "labels.npy",
        emb_root / "filenames.npy",
        emb_root / "classnames.npy",
    ]
    missing_embeddings = [path.name for path in required_embedding_files if not path.exists()]

    expected_splits = ["train", "val", "test"]
    split_dirs = [data_root / split for split in expected_splits]
    split_status = [path.exists() and path.is_dir() for path in split_dirs]
    image_count = sum(len(list(path.rglob("*.jpg"))) for path in split_dirs if path.exists())
    data_ok = all(split_status) and image_count > 0
    if data_ok:
        data_detail = f"{image_count} image(s) détectée(s) dans train/val/test"
    else:
        missing_splits = [split for split, ok in zip(expected_splits, split_status) if not ok]
        if missing_splits:
            data_detail = f"dossiers manquants: {', '.join(missing_splits)}"
        else:
            data_detail = "aucune image .jpg détectée dans data/out"

    missing_embedding_paths_count = 0
    embeddings_state = "Valide"
    embeddings_card_ok = not missing_embeddings
    embeddings_detail = (
        f"{len(required_embedding_files)} fichiers requis présents"
        if embeddings_card_ok
        else f"manquants: {', '.join(missing_embeddings)}"
    )

    filenames_path = emb_root / "filenames.npy"
    if embeddings_card_ok and data_ok and filenames_path.exists():
        stored_filenames = np.load(filenames_path, allow_pickle=True)
        missing_embedding_paths_count = 0
        for fp in stored_filenames:
            resolved = resolve_stored_path(fp)
            if not resolved.exists():
                missing_embedding_paths_count += 1

        if missing_embedding_paths_count > 0:
            embeddings_card_ok = False
            embeddings_state = "Désynchronisé"
            embeddings_detail = (
                f"{missing_embedding_paths_count} chemin(s) d'embeddings ne pointent plus "
                "vers un fichier présent dans data/out"
            )

    rows = [
        status_row("Modèle", model_ok, model_detail),
        status_row("Embeddings", embeddings_card_ok, embeddings_detail, state=embeddings_state),
        status_row("data/out", data_ok, data_detail),
    ]

    return {
        "rows": rows,
        "upload_enabled": bool(model_ok and data_ok and missing_embedding_paths_count == 0 and not missing_embeddings),
        "missing_embedding_paths_count": missing_embedding_paths_count,
    }


def render_runtime_status() -> dict[str, object]:
    """
    Affiche un panneau déroulant discret pour l'état des ressources du projet.
    """
    status = _inspect_runtime_assets()
    rows = status["rows"]

    has_error = not bool(status["upload_enabled"])
    button_bg = "#fdecec" if has_error else "#eef6ff"
    button_fg = "#b42318" if has_error else "#175cd3"
    button_border = "#f04438" if has_error else "#84caff"
    #status_icon = "🔴" if has_error else "🟢"
    #button_label = f"État des ressources {status_icon}"
    button_label = "État des ressources"

    st.markdown(
        f"""
        <style>
        div[data-testid="stExpander"] details {{
            border: 1px solid {button_border};
            border-radius: 12px;
            overflow: hidden;
        }}
        div[data-testid="stExpander"] details summary {{
            background: {button_bg};
            color: {button_fg};
            font-weight: 700;
        }}
        div[data-testid="stExpander"] details summary:hover {{
            background: {button_bg};
            color: {button_fg};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(button_label, expanded=False):
        cols = st.columns(len(rows))
        for col, row in zip(cols, rows):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        border-left: 6px solid {row['accent']};
                        background: {row['tone']};
                        border-radius: 12px;
                        padding: 0.85rem 1rem;
                        min-height: 128px;
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.35rem;">
                            {row['label']}
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: {row['accent']}; margin-bottom: 0.35rem;">
                            {row['state']}
                        </div>
                        <div style="font-size: 0.92rem; line-height: 1.35;">
                            {row['detail']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if status["missing_embedding_paths_count"]:
            st.warning(
                "Les embeddings sont désynchronisés avec `data/out`. "
                "L'upload est bloqué tant que les embeddings n'ont pas été régénérés."
            )

    return status


# =============================================================================
# INITIALISATION DES RESSOURCES
# =============================================================================

retriever = None
retriever_error = None

try:
    retriever = get_retriever()
except Exception as exc:
    retriever_error = exc
# Instancie (ou récupère depuis le cache) le moteur de recherche.
# Cette ligne est exécutée lors du rendu du script Streamlit.
# Dans Streamlit, le script est relancé de haut en bas à chaque interaction utilisateur.
# Grâce au cache, cette ré-exécution ne reconstruit pas inutilement le moteur.

latent_bundle = None
# Variable initialisée par défaut à None.
# Elle contiendra plus tard soit les données UMAP, soit restera vide si indisponibles.

latent_error = None
# Variable destinée à stocker une éventuelle exception rencontrée
# lors du chargement des données UMAP.

try:
    latent_bundle = load_latent_and_meta()
    # Tente de charger les données UMAP.
    # Si tout se passe bien, latent_bundle contiendra un tuple prêt à être utilisé
    # dans la section de visualisation plus bas.
except Exception as exc:
    latent_error = exc
    # En cas d'erreur, on la stocke sans faire crasher l'application.
    # Cela permet à l'interface principale de rester utilisable même si l'UMAP est cassé.
    # C'est un choix ergonomique important : une fonctionnalité secondaire défaillante
    # ne doit pas rendre inutilisable le cœur du produit.


# =============================================================================
# INTERFACE UTILISATEUR - CONTRÔLES D'ENTRÉE
# =============================================================================

runtime_status = render_runtime_status()

if retriever_error is not None:
    st.warning(f"Moteur de retrieval indisponible : {retriever_error}")

upload_disabled = (retriever_error is not None) or (not runtime_status["upload_enabled"])

uploaded = st.file_uploader(
    "Upload une image (jpg/png/webp)",
    type=["jpg", "jpeg", "png", "webp"],
    disabled=upload_disabled,
)
# Affiche un composant de téléchargement de fichier.
# L'utilisateur peut envoyer une image dans l'un des formats autorisés.
# La variable 'uploaded' contiendra soit None (si rien n'est chargé),
# soit un objet UploadedFile fourni par Streamlit.

k = 4
# Nombre de résultats similaires à demander au moteur.
# Ici, on fixe top-k à 4 de manière statique.
# Une version plus avancée pourrait exposer ce paramètre à l'utilisateur via
# un slider Streamlit, mais le garder fixe simplifie l'expérience.

show_gradcam = st.checkbox(
    "Afficher Grad-CAM++ (top-1)",
    value=False,
)
# Ajoute une case à cocher pour activer ou non la visualisation Grad-CAM++.
# value=False signifie qu'elle est décochée par défaut.
# Cette option déclenche potentiellement un calcul coûteux, d'où son caractère optionnel.

show_gradcam_history = False
if show_gradcam:
    show_gradcam_history = st.checkbox(
        "Grad-CAM history",
        value=False,
    )

available_layers: list[str] = []
if show_gradcam and retriever is not None:
    available_layers = retriever.available_explanation_layers()
    if not available_layers:
        st.warning("Aucune couche compatible n'a été trouvée pour Grad-CAM++.")


# =============================================================================
# TRAITEMENT PRINCIPAL
# =============================================================================

if uploaded is not None and retriever is not None:
    # Toute la logique principale de recherche est exécutée seulement
    # si l'utilisateur a effectivement uploadé une image.
    # Cette condition joue le rôle de "point d'entrée utilisateur" du workflow.
    # Tant qu'aucune image n'est fournie, l'application reste dans un état d'attente.

    suffix = Path(uploaded.name).suffix if uploaded.name else ".jpg"
    # On récupère l'extension du fichier uploadé (.jpg, .png, etc.)
    # pour la réutiliser dans le fichier temporaire.
    # Si uploaded.name est absent, on prend .jpg par défaut.

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        # Crée un fichier temporaire sur disque.
        # delete=False signifie que le fichier ne sera pas supprimé automatiquement
        # à la fermeture du contexte.
        # C'est utile si le moteur de retrieval attend un chemin de fichier réel.
        f.write(uploaded.read())
        # Lit le contenu binaire du fichier uploadé puis l'écrit dans le fichier temporaire.

        query_path = f.name
        # Stocke le chemin du fichier temporaire.
        # Ce chemin sera passé au moteur de recherche.
        # On passe donc d'un objet Streamlit en mémoire à un fichier concret sur disque,
        # ce qui facilite l'intégration avec du code existant orienté "path de fichier".

    with st.spinner("Recherche des œuvres similaires..."):
        results = retriever.top_k_similar(query_path, k=k)
        # Affiche une animation d'attente pendant l'exécution de la recherche.
        # top_k_similar est supposée renvoyer une liste de résultats,
        # chaque résultat étant probablement un dictionnaire contenant au moins :
        # - filepath
        # - style
        # - similarity

    if not results:
        st.error("Aucun résultat n'a été retourné par le moteur de recherche.")
        # Affiche un message d'erreur utilisateur si la liste est vide.

        st.stop()
        # Arrête immédiatement l'exécution du script Streamlit pour cette interaction.
        # Cela évite d'accéder à results[0] plus bas et donc de provoquer une erreur.

    best = results[0]
    # Récupère le premier résultat, supposé être le plus pertinent.
    # On l'utilise ensuite comme top-1 pour le style suggéré, Grad-CAM++ et le highlight UMAP.
    # Le reste de l'interface s'organise donc autour de ce meilleur voisin :
    # c'est lui qui sert de référence principale pour l'explication visuelle.
    second_best = results[1] if len(results) > 1 else None
    # Conserve aussi le top-2 quand il existe pour pouvoir le pointer dans l'UMAP.
    third_best = results[2] if len(results) > 2 else None
    # Conserve aussi le top-3 quand il existe pour l'afficher dans l'UMAP.

    # -------------------------------------------------------------------------
    # Affichage de l'image source
    # -------------------------------------------------------------------------
    st.subheader("Image source")
    # Affiche un sous-titre pour introduire la section de l'image requête.

    st.image(
        query_path,
        caption="Image requête",
        # width="stretch",
        width=600,
    )
    # Affiche l'image uploadée par l'utilisateur.
    # query_path est le chemin local temporaire de l'image.
    # caption ajoute une légende sous l'image.
    # width=600 fixe une largeur d'affichage en pixels.
    # La ligne width="stretch" est commentée : elle aurait étiré l'image selon l'espace dispo.

    source_artist, source_title = _extract_artist_and_title(uploaded.name or query_path)
    st.markdown(
        f"""
        <p style="line-height:1.1; margin:0;">
            <strong>{source_artist}</strong> <br>
            <em>{source_title}</em>
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"**Style suggéré :** {best['style']}")
    # Affiche le style du meilleur résultat en gras grâce au Markdown.
    # best['style'] est supposé provenir du moteur de retrieval.

    # -------------------------------------------------------------------------
    # Affichage des résultats visuels
    # -------------------------------------------------------------------------
    st.subheader("Comparaison visuelle")
    # Sous-titre de la section qui montre les œuvres similaires.

    cols = st.columns(min(4, len(results)))
    # Crée un ensemble de colonnes Streamlit pour afficher les résultats côte à côte.
    # min(4, len(results)) permet de ne pas créer plus de colonnes qu'il n'y a de résultats,
    # tout en limitant l'affichage à 4 colonnes maximum.

    for i, res in enumerate(results):
        # Parcourt chaque résultat avec son indice i.
        # `enumerate` est utile ici car on a besoin à la fois :
        # - de la donnée `res`,
        # - et du rang `i` pour afficher "Top 1", "Top 2", etc.

        artist, title = _extract_artist_and_title(res["filepath"])
        # Tente d'extraire artiste et titre depuis le nom de fichier du résultat.

        with cols[i % len(cols)]:
            # Choisit la colonne dans laquelle afficher ce résultat.
            # Le modulo permet de répartir les cartes dans les colonnes disponibles.
            # Ici, comme le nombre de résultats est k=4, cela correspond souvent à une colonne par résultat.

            st.image(
                res["filepath"],
                caption=f"Top {i + 1} — {res['style']} ({res['similarity']:.3f})",
                width="stretch",
            )
            # Affiche l'image candidate.
            # res["filepath"] est son chemin.
            # La légende montre :
            # - son rang (Top 1, Top 2, ...)
            # - son style
            # - son score de similarité formaté à 3 décimales.
            # width="stretch" demande à Streamlit d'occuper la largeur du conteneur.

            st.markdown(
                f"""
                <p style="line-height:1.1; margin:0;">
                    <strong>{artist}</strong> <br>
                    <em>{title}</em>
                </p>
                """,
                unsafe_allow_html=True
            )
            # Affiche en HTML l'artiste et le titre juste sous l'image.
            # <strong> met le nom de l'artiste en gras.
            # <em> met le titre en italique.
            # line-height réduit l'espacement vertical.
            # margin:0 évite des marges inutiles.
            # unsafe_allow_html=True autorise l'interprétation HTML par Streamlit.
            # Il faut l'utiliser avec prudence, mais ici le contenu est simple et maîtrisé.

    # -------------------------------------------------------------------------
    # Tableau récapitulatif
    # -------------------------------------------------------------------------
    st.subheader("Résumé des résultats")
    # Introduit la section tabulaire.

    rows = []
    # Cette liste va contenir un dictionnaire par résultat.
    # Elle sera ensuite convertie en DataFrame Pandas.

    for i, res in enumerate(results):
        artist, title = _extract_artist_and_title(res["filepath"])
        # Réutilise l'extraction artiste/titre pour enrichir le tableau.
        # Cela évite de dupliquer une logique de parsing ailleurs dans le code
        # et garantit une présentation cohérente entre cartes visuelles et tableau.

        rows.append(
            {
                "rang": i + 1,
                # Position du résultat dans le classement.

                "artiste": artist,
                # Nom de l'artiste extrait du nom de fichier.

                "tableau": title,
                # Titre de l'œuvre extrait du nom de fichier.

                "style": res["style"],
                # Style associé au résultat.

                "similarité": round(float(res["similarity"]), 4),
                # Score de similarité converti explicitement en float puis arrondi à 4 décimales.
                # Le float(...) garantit un type affichable même si similarity est un scalaire NumPy.

                "fichier": Path(str(res["filepath"])).name,
                # Ne garde que le nom du fichier (sans le dossier).

                "chemin": str(res["filepath"]),
                # Garde aussi le chemin complet, utile pour debug ou inspection.
            }
        )

    df_results = pd.DataFrame(rows)
    # Transforme la liste de dictionnaires en tableau structuré Pandas.

    st.dataframe(df_results, width="stretch", hide_index=True)
    # Affiche le DataFrame dans Streamlit.
    # width="stretch" utilise toute la largeur disponible.
    # hide_index=True masque l'index par défaut de Pandas, peu utile ici.

    # -------------------------------------------------------------------------
    # Visualisation UMAP
    # -------------------------------------------------------------------------
    if latent_error is not None:
        st.warning(f"Visualisation UMAP indisponible : {latent_error}")
        # Si le chargement UMAP a échoué plus tôt, on affiche l'erreur ici.

    elif latent_bundle is not None:
        st.subheader("Espace latent (UMAP interactif)")
        # Si les données sont disponibles, on affiche la section UMAP.

        latent_2d, labels, classnames, filenames = latent_bundle
        # Décompacte les 4 éléments retournés par load_latent_and_meta().

        style_names = _build_style_names(labels, classnames)
        # Convertit les labels numériques en noms de styles lisibles.
        # C'est un point pédagogique important :
        # les modèles et fichiers intermédiaires manipulent volontiers des identifiants numériques,
        # alors que l'interface doit toujours afficher des informations interprétables humainement.

        short_filenames = [Path(str(fp)).name for fp in filenames]
        # Crée une version courte des noms de fichiers, sans les chemins,
        # pour un affichage plus propre dans les tooltips.

        artists = []
        titles = []
        # Prépare deux listes qui contiendront artiste et titre pour chaque point UMAP.

        for fp in filenames:
            artist, title = _extract_artist_and_title(str(fp))
            # Extrait artiste et titre depuis chaque nom de fichier.

            artists.append(artist)
            titles.append(title)
            # Remplit les listes pour les injecter ensuite dans le DataFrame.

        similarity_by_filepath = {
            str(res["filepath"]): f"{float(res['similarity']) * 100:.1f}%"
            for res in results
        }
        # Associe chaque résultat retourné à son score de similarité formaté en pourcentage.
        # Les autres points de l'UMAP n'ont pas de score lié à la requête courante.

        tooltip_html = []
        for fp, style, artist, title in zip(filenames, style_names, artists, titles):
            similarity = similarity_by_filepath.get(str(fp))
            lines = [
                f"<b>Style:</b> {style}",
                f"<b>Artiste:</b> {artist}",
                f"<b>Tableau:</b> {title}",
            ]
            if similarity:
                lines.append(f"<b>Similarité:</b> {similarity}")
            tooltip_html.append("<br>".join(lines))
        # Construit un contenu de tooltip sur mesure pour n'afficher la similarité
        # que sur les points concernés par la requête courante.

        df_umap = pd.DataFrame(
            {
                "x": latent_2d[:, 0],
                # Première coordonnée UMAP de chaque point.

                "y": latent_2d[:, 1],
                # Deuxième coordonnée UMAP de chaque point.

                "Label": labels,
                # Label brut de classe.

                "Style": style_names,
                # Nom lisible du style.

                "Artiste": artists,
                # Artiste extrait du nom de fichier.

                "Tableau": titles,
                # Titre de l'œuvre.

                "Fichier": short_filenames,
                # Nom de fichier court.

                "tooltip_html": tooltip_html,
                # Tooltip HTML complet utilisé par Plotly pour chaque point.

                "filepath": [str(fp) for fp in filenames],
                # Chemin complet, potentiellement utile pour retrouver un point précis.
            }
        )
        # On rassemble ici toutes les données utiles dans un DataFrame unique.
        # Ce choix simplifie beaucoup la suite, car Plotly Express et les filtres Pandas
        # travaillent très naturellement avec ce format tabulaire.
        # En d'autres termes : on convertit des tableaux NumPy "techniques"
        # en structure "métier / interface" plus facile à manipuler.

        styles_disponibles = sorted(df_umap["Style"].astype(str).unique().tolist())
        # Récupère la liste unique des styles présents dans le DataFrame,
        # convertie en chaînes, puis triée alphabétiquement.
        # Cela sert de base aux filtres interactifs.

        styles_selectionnes = st.multiselect(
            "Filtrer les styles affichés dans l'UMAP",
            options=styles_disponibles,
            default=styles_disponibles,
        )
        # Affiche une sélection multiple permettant à l'utilisateur de choisir
        # quels styles afficher.
        # Par défaut, tous les styles sont sélectionnés.

        df_umap_filtered = df_umap[df_umap["Style"].isin(styles_selectionnes)].copy()
        # Filtre les points UMAP selon les styles choisis.
        # .copy() évite certains avertissements Pandas liés aux vues/slices.
        # C'est aussi plus sûr si l'on souhaite modifier ensuite le DataFrame filtré
        # sans impacter accidentellement l'original.

        if df_umap_filtered.empty:
            st.warning("Aucun point à afficher : aucun style sélectionné.")
            # Si l'utilisateur désélectionne tout, on l'indique explicitement.
        else:
            fig = px.scatter(
                df_umap_filtered,
                x="x",
                y="y",
                color="Style",
                hover_data={
                    "x": False,
                    "y": False,
                    "Label": False,
                    "Style": False,
                    "Artiste": False,
                    "Tableau": False,
                    "Fichier": False,
                    "tooltip_html": False,
                    "filepath": False,
                },
                custom_data=["tooltip_html"],
                opacity=0.45,
                title="Projection UMAP des embeddings",
            )
            # Crée un nuage de points interactif Plotly.
            # color="Style" colore les points par style.
            # hover_data permet de choisir précisément quelles colonnes apparaissent au survol.
            # x, y, Label, filepath, etc. peuvent être masqués pour garder un tooltip lisible.
            # opacity=0.45 rend les points semi-transparents, utile quand ils se chevauchent.
            # Plotly Express est très pratique ici parce qu'il permet de produire
            # rapidement une visualisation riche à partir d'un DataFrame et de quelques arguments.

            fig.update_traces(
                marker=dict(size=7),
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
            # Réduit / fixe la taille des marqueurs pour améliorer la lisibilité du nuage.

            upload_anchor_points = []
            for res in results[:3]:
                idx_res = _find_best_index(filenames, str(res["filepath"]))
                if idx_res is None:
                    continue
                weight = max(float(res["similarity"]), 0.0)
                upload_anchor_points.append(
                    (
                        float(latent_2d[idx_res, 0]),
                        float(latent_2d[idx_res, 1]),
                        weight,
                    )
                )

            if upload_anchor_points:
                total_weight = sum(weight for _, _, weight in upload_anchor_points)
                if total_weight <= 0:
                    total_weight = float(len(upload_anchor_points))
                    upload_anchor_points = [(x, y, 1.0) for x, y, _ in upload_anchor_points]

                x_upload = sum(x * weight for x, _, weight in upload_anchor_points) / total_weight
                y_upload = sum(y * weight for _, y, weight in upload_anchor_points) / total_weight

                fig.add_trace(
                    go.Scatter(
                        x=[x_upload],
                        y=[y_upload],
                        mode="markers+text",
                        name="Tableau uploadé",
                        text=["Upload"],
                        textposition="top left",
                        marker=dict(
                            size=17,
                            symbol="star",
                            color="#d92d20",
                            line=dict(width=1.5, color="white"),
                        ),
                        hovertemplate=(
                            "<b>Tableau uploadé</b><br>"
                            f"Artiste: {source_artist}<br>"
                            f"Tableau: {source_title}<br>"
                            "Position estimée depuis les top résultats"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

                fig.add_annotation(
                    x=x_upload,
                    y=y_upload,
                    text="Upload",
                    showarrow=True,
                    arrowhead=2,
                    ax=28,
                    ay=-16,
                    bgcolor="#fef3f2",
                    bordercolor="#d92d20",
                    borderwidth=1,
                )
            # Le point Upload est estimé à partir des meilleurs voisins déjà projetés.

            idx_best = _find_best_index(filenames, str(best["filepath"]))
            # Tente de retrouver le point UMAP correspondant au meilleur résultat top-1.

            if idx_best is not None:
                x_best = latent_2d[idx_best, 0]
                # Coordonnée x du meilleur point.

                y_best = latent_2d[idx_best, 1]
                # Coordonnée y du meilleur point.

                best_filename = Path(str(best["filepath"])).name
                # Nom de fichier du meilleur résultat, sans son chemin.

                best_artist, best_title = _extract_artist_and_title(best["filepath"])
                # Extrait artiste et titre pour enrichir le tooltip du point mis en évidence.

                fig.add_trace(
                    go.Scatter(
                        x=[x_best],
                        y=[y_best],
                        mode="markers+text",
                        name="Top-1 sélectionné",
                        text=[best["style"]],
                        textposition="top center",
                        marker=dict(
                            size=18,
                            symbol="circle-open",
                            line=dict(width=3, color="black"),
                        ),
                        hovertemplate=(
                            "<b>Top-1 sélectionné</b><br>"
                            f"Style: {best['style']}<br>"
                            f"Artiste: {best_artist}<br>"
                            f"Tableau: {best_title}<br>"
                            f"Fichier: {best_filename}<br>"
                            f"Similarité: {float(best['similarity']) * 100:.1f}%"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
                # Ajoute une nouvelle trace Plotly pour surligner visuellement le top-1.
                # mode="markers+text" affiche à la fois un point et un texte.
                # symbol="circle-open" dessine un cercle vide autour du point.
                # line=dict(...) définit le contour noir épais.
                # hovertemplate personnalise complètement le contenu du tooltip.
                # <extra></extra> supprime le petit encart supplémentaire Plotly souvent peu utile.
                # On passe ici de Plotly Express (très pratique pour la base du graphique)
                # à Plotly Graph Objects (plus bas niveau) afin d'ajouter une surcouche
                # vraiment sur mesure.

                fig.add_annotation(
                    x=x_best,
                    y=y_best,
                    text="Top 1",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )
                # Ajoute une annotation textuelle avec flèche pointant vers le meilleur résultat.
                # ax et ay contrôlent le décalage de l'étiquette par rapport au point.
                # bgcolor, bordercolor et borderwidth améliorent la lisibilité de l'annotation.

            if second_best is not None:
                idx_second = _find_best_index(filenames, str(second_best["filepath"]))
                # Tente de retrouver le point UMAP correspondant au deuxième meilleur résultat.

                if idx_second is not None:
                    x_second = latent_2d[idx_second, 0]
                    y_second = latent_2d[idx_second, 1]
                    second_filename = Path(str(second_best["filepath"])).name
                    second_artist, second_title = _extract_artist_and_title(second_best["filepath"])

                    fig.add_trace(
                        go.Scatter(
                            x=[x_second],
                            y=[y_second],
                            mode="markers+text",
                            name="Top-2 sélectionné",
                            text=[second_best["style"]],
                            textposition="bottom center",
                            marker=dict(
                                size=16,
                                symbol="diamond-open",
                                line=dict(width=3, color="#b54708"),
                            ),
                            hovertemplate=(
                                "<b>Top-2 sélectionné</b><br>"
                                f"Style: {second_best['style']}<br>"
                                f"Artiste: {second_artist}<br>"
                                f"Tableau: {second_title}<br>"
                                f"Fichier: {second_filename}<br>"
                                f"Similarité: {float(second_best['similarity']) * 100:.1f}%"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                    fig.add_annotation(
                        x=x_second,
                        y=y_second,
                        text="Top 2",
                        showarrow=True,
                        arrowhead=2,
                        ax=-24,
                        ay=34,
                        bgcolor="#fff7ed",
                        bordercolor="#b54708",
                        borderwidth=1,
                    )
                    # Ajoute un second repère, visuellement distinct du top-1.

            if third_best is not None:
                idx_third = _find_best_index(filenames, str(third_best["filepath"]))
                # Tente de retrouver le point UMAP correspondant au troisième meilleur résultat.

                if idx_third is not None:
                    x_third = latent_2d[idx_third, 0]
                    y_third = latent_2d[idx_third, 1]
                    third_filename = Path(str(third_best["filepath"])).name
                    third_artist, third_title = _extract_artist_and_title(third_best["filepath"])

                    fig.add_trace(
                        go.Scatter(
                            x=[x_third],
                            y=[y_third],
                            mode="markers+text",
                            name="Top-3 sélectionné",
                            text=[third_best["style"]],
                            textposition="middle right",
                            marker=dict(
                                size=15,
                                symbol="square-open",
                                line=dict(width=3, color="#175cd3"),
                            ),
                            hovertemplate=(
                                "<b>Top-3 sélectionné</b><br>"
                                f"Style: {third_best['style']}<br>"
                                f"Artiste: {third_artist}<br>"
                                f"Tableau: {third_title}<br>"
                                f"Fichier: {third_filename}<br>"
                                f"Similarité: {float(third_best['similarity']) * 100:.1f}%"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                    fig.add_annotation(
                        x=x_third,
                        y=y_third,
                        text="Top 3",
                        showarrow=True,
                        arrowhead=2,
                        ax=34,
                        ay=18,
                        bgcolor="#eff8ff",
                        bordercolor="#175cd3",
                        borderwidth=1,
                    )
                    # Ajoute un troisième repère distinct pour le top-3.

            fig.update_layout(
                xaxis_title="Dimension UMAP 1",
                yaxis_title="Dimension UMAP 2",
                legend_title="Styles",
                height=700,
                hoverlabel=dict(
                    font=dict(size=16),
                ),
            )
            # Personnalise la mise en page du graphique :
            # titres d'axes, titre de légende et hauteur globale du composant.

            st.plotly_chart(fig, width="stretch")
            # Affiche le graphique interactif dans Streamlit.
            # width="stretch" permet au graphique d'occuper toute la largeur disponible.

            st.caption(
                "Chaque point représente une œuvre projetée dans un espace latent 2D. "
                "Les couleurs correspondent aux styles artistiques. "
                "Les points annotés correspondent au tableau uploadé ainsi qu'aux meilleurs résultats top-1, top-2 et top-3."
            )
            # Ajoute une légende explicative sous le graphique pour aider l'utilisateur
            # à interpréter la visualisation.

            st.caption(
                "Note : la proximité visuelle dans l'UMAP ne correspond pas toujours exactement "
                "au classement de similarité, car la projection 2D déforme partiellement "
                "les distances calculées dans l'espace latent d'origine."
            )

    # -------------------------------------------------------------------------
    # Explication visuelle avec Grad-CAM++
    # -------------------------------------------------------------------------
    if show_gradcam and available_layers:
        # Ce bloc n'est exécuté que si l'utilisateur a coché la case correspondante.
        # On garde ce calcul optionnel car Grad-CAM++ peut être plus lent
        # que la simple recherche des voisins les plus proches.

        st.subheader("Explication visuelle (Grad-CAM++)")
        # Titre de la section d'explicabilité.

        try:
            primary_layer_numbers = [245]
            history_layer_numbers = [10, 20, 50, 70, 100, 120, 150, 200, 240, 245]

            primary_layers, missing_primary_layers = _select_explanation_layers(
                available_layers,
                primary_layer_numbers,
            )
            history_layers, missing_history_layers = _select_explanation_layers(
                available_layers,
                history_layer_numbers,
            )
            _, layer_labels = _format_explanation_layer_options(available_layers)

            if not primary_layers:
                st.warning("La couche 245 n'est pas disponible pour ce modèle.")
                st.stop()

            with st.spinner("Calcul des cartes Grad-CAM++..."):
                primary_explanations = [
                    (
                        layer_number,
                        retriever.explain_similarity(
                            query_path,
                            best["filepath"],
                            target_layer_name=layer_name,
                        ),
                    )
                    for layer_number, layer_name in primary_layers
                ]
                # Calcule l'explication principale pour la couche 245.

                history_explanations = []
                if show_gradcam_history:
                    history_explanations = [
                        (
                            layer_number,
                            retriever.explain_similarity(
                                query_path,
                                best["filepath"],
                                target_layer_name=layer_name,
                            ),
                        )
                        for layer_number, layer_name in history_layers
                    ]
                    # Calcule l'historique complet uniquement si l'utilisateur le demande.

            best_artist, best_title = _extract_artist_and_title(best["filepath"])
            # Extrait l'artiste et le titre du meilleur résultat pour les afficher textuellement.

            st.markdown(
                f"""
                **Top-1 sélectionné :**
                **Artiste :** {best_artist}
                **Tableau :** {best_title}
                """
            )
            # Affiche des informations textuelles sur le top-1.
            # À noter : en Markdown classique, des sauts de ligne plus explicites
            # ou des puces pourraient encore améliorer le rendu.

            st.caption("Affichage principal : image uploadée vs top-1 à la couche 245.")

            if missing_primary_layers:
                missing_text = ", ".join(str(layer_number) for layer_number in missing_primary_layers)
                st.caption(f"Couches principales non disponibles : {missing_text}.")

            for layer_number, explanation in primary_explanations:
                layer_name = explanation["target_layer"]
                layer_label = layer_labels.get(layer_name, layer_name)
                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        explanation["query_overlay"],
                        caption=(
                            f"Upload ({explanation['method']}) • couche {layer_number} • "
                            f"{layer_label} • `{layer_name}` • similarité {explanation['similarity']:.3f}"
                        ),
                        width="stretch",
                    )

                with c2:
                    st.image(
                        explanation["candidate_overlay"],
                        caption=(
                            f"Top-1 ({explanation['method']}) • couche {layer_number} • "
                            f"{layer_label} • `{layer_name}` • similarité {explanation['similarity']:.3f}"
                        ),
                        width="stretch",
                    )
                # Affiche côte à côte la requête et le top-1 pour chaque couche demandee.

            if show_gradcam_history:
                st.markdown("**Grad-CAM history**")
                st.caption(
                    f"{len(history_explanations)} paires de cartes affichees pour les couches "
                    "10, 20, 50, 70, 100, 120, 150, 200, 240, 245."
                )

                if missing_history_layers:
                    missing_text = ", ".join(str(layer_number) for layer_number in missing_history_layers)
                    st.caption(f"Couches d'historique non disponibles pour ce modèle : {missing_text}.")

                for layer_number, explanation in history_explanations:
                    layer_name = explanation["target_layer"]
                    layer_label = layer_labels.get(layer_name, layer_name)
                    c1, c2 = st.columns(2)

                    with c1:
                        st.image(
                            explanation["query_overlay"],
                            caption=(
                                f"Upload ({explanation['method']}) • couche {layer_number} • "
                                f"{layer_label} • `{layer_name}` • similarité {explanation['similarity']:.3f}"
                            ),
                            width="stretch",
                        )

                    with c2:
                        st.image(
                            explanation["candidate_overlay"],
                            caption=(
                                f"Top-1 ({explanation['method']}) • couche {layer_number} • "
                                f"{layer_label} • `{layer_name}` • similarité {explanation['similarity']:.3f}"
                            ),
                            width="stretch",
                        )

        except Exception as exc:
            st.warning(f"Grad-CAM++ indisponible : {exc}")
            # Si le calcul Grad-CAM++ échoue, on n'interrompt pas toute l'application.
            # On affiche simplement un avertissement explicatif.

    else:
        st.info(
            "Active l'option Grad-CAM++ pour visualiser les zones qui "
            "contribuent à la similarité du top-1."
        )
        # Si l'option n'est pas activée, on affiche un message informatif pour guider l'utilisateur.

elif uploaded is None:
    if upload_disabled:
        st.info("Upload désactivé tant que le modèle, les embeddings et `data/out` ne sont pas synchronisés.")
    else:
        st.info("Charge une image pour lancer la recherche.")
    # Cas initial : aucun fichier n'a encore été uploadé.
    # On affiche simplement une consigne à l'utilisateur.
    # Ce `else` correspond à l'état d'accueil de l'application :
    # l'interface est chargée, les ressources sont prêtes, mais aucun workflow
    # de recherche n'a encore été déclenché.
else:
    st.error("Le moteur n'est pas prêt. Vérifie le modèle, les embeddings et data/out.")
