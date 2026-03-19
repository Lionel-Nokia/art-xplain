"""
================================================================================
Art-Xplain - Application Streamlit de recherche de similarité stylistique
================================================================================

Objectif
--------
Cette application permet à un utilisateur de charger une image d'œuvre d'art
(ou d'image assimilée), puis de rechercher dans une base d'images les œuvres
les plus proches visuellement / stylistiquement.

Fonctionnalités principales
---------------------------
1. Upload d'une image utilisateur
2. Recherche des k œuvres les plus similaires
3. Affichage de l'image requête
4. Affichage des résultats visuels avec score de similarité
5. Affichage d'un tableau récapitulatif
6. Affichage optionnel d'une explication Grad-CAM sur le meilleur résultat
7. Affichage optionnel d'une projection UMAP interactive des embeddings

Architecture logique
--------------------
L'application repose sur deux briques principales :

- Une interface Streamlit
  -> gère l'upload, l'affichage et les interactions utilisateur

- Un moteur métier (StyleRetriever)
  -> gère la recherche d'images similaires
  -> gère éventuellement l'explication Grad-CAM
  -> encapsule la logique ML / retrieval

Dépendances importantes
-----------------------
- Streamlit : interface web
- NumPy : tableaux numériques
- Pandas : tableau récapitulatif des résultats
- Plotly : visualisation interactive UMAP
- src.utils.load_config : chargement de la configuration projet
- src.retrieval.StyleRetriever : moteur de recherche stylistique

Convention
----------
Ce fichier cherche à être :
- lisible
- pédagogique
- robuste
- facile à maintenir

Les commentaires ci-dessous expliquent :
- ce que fait chaque bloc
- pourquoi il est présent
- ce qu'il attend en entrée
- ce qu'il produit en sortie
- les cas limites éventuels
================================================================================
"""

from __future__ import annotations

# =============================================================================
# IMPORTS STANDARDS PYTHON
# =============================================================================
# sys
# ----
# Utilisé ici pour modifier dynamiquement le chemin de recherche des modules
# Python (sys.path). Cela permet d'importer les modules internes du projet
# même si le script est lancé depuis un autre répertoire.
import sys

# tempfile
# --------
# Permet de créer un fichier temporaire sur disque.
# C'est utile car l'image chargée via Streamlit est en mémoire, alors que
# le moteur de retrieval semble attendre un chemin de fichier.
import tempfile

# Path
# ----
# Classe moderne de manipulation de chemins (pathlib).
# Plus robuste et plus lisible que les concaténations de chaînes.
from pathlib import Path

# =============================================================================
# IMPORTS DATA / VISUALISATION / INTERFACE
# =============================================================================
# numpy
# -----
# Utilisé pour :
# - convertir / valider des tableaux
# - charger les embeddings et métadonnées
# - retrouver l'indice d'un fichier
import numpy as np

# pandas
# ------
# Utilisé pour construire un tableau récapitulatif des résultats de similarité.
import pandas as pd

# plotly.express
# --------------
# Utilisé pour construire rapidement un scatter plot interactif UMAP.
import plotly.express as px

# plotly.graph_objects
# --------------------
# Utilisé pour ajouter des traces personnalisées au graphique Plotly,
# en particulier pour surligner le meilleur résultat (top-1).
import plotly.graph_objects as go

# streamlit
# ---------
# Framework principal de l'application web interactive.
import streamlit as st

# =============================================================================
# GESTION DES IMPORTS INTERNES AU PROJET
# =============================================================================
# But
# ---
# Garantir que le dossier racine du projet est visible dans le PYTHONPATH,
# afin de pouvoir importer les modules internes situés sous "src/".
#
# Pourquoi c'est nécessaire
# -------------------------
# Selon la manière dont Streamlit lance le script, le répertoire courant n'est
# pas toujours celui attendu. Sans cette adaptation, l'import :
#
#     from src.utils import load_config
#
# pourrait échouer avec un ModuleNotFoundError.
#
# Hypothèse de structure de projet
# --------------------------------
# projet/
# ├── src/
# │   ├── utils.py
# │   └── retrieval.py
# └── app/
#     └── streamlit_app.py
#
# Ici, on suppose que ce fichier est situé dans un sous-dossier du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# On ajoute la racine du projet au sys.path si elle n'y figure pas déjà.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import d'une fonction utilitaire de chargement de configuration.
# Cette configuration contient probablement des chemins utiles
# (ex : dossier des embeddings, modèles, ressources, etc.).
from src.utils import load_config

# Import du moteur principal de recherche visuelle / stylistique.
# Cette classe constitue le cœur métier de l'application.
from src.retrieval import StyleRetriever


# =============================================================================
# CONFIGURATION GÉNÉRALE DE L'INTERFACE STREAMLIT
# =============================================================================
# But
# ---
# Définir l'apparence générale de la page et le titre principal affiché.
#
# layout="wide"
# -------------
# Permet d'utiliser une largeur plus importante, utile pour :
# - l'affichage côte à côte des images
# - l'affichage de l'UMAP
# - l'affichage d'un tableau large
st.set_page_config(page_title="Art-Xplain", layout="wide")

# Titre principal visible en haut de l'application.
st.title("Art-Xplain — Similarité stylistique")


# =============================================================================
# CHARGEMENT DU MOTEUR DE RETRIEVAL
# =============================================================================
@st.cache_resource
def get_retriever() -> StyleRetriever:
    """
    Crée et retourne l'objet StyleRetriever.

    But
    ---
    Initialiser une seule fois le moteur de recherche, même si Streamlit
    relance le script à chaque interaction utilisateur.

    Pourquoi utiliser @st.cache_resource ?
    -------------------------------------
    Dans Streamlit, chaque interaction (checkbox, multiselect, upload, etc.)
    peut provoquer un rerun complet du script.

    Si StyleRetriever charge :
    - un modèle de deep learning,
    - des embeddings,
    - un index de recherche,
    - un backend lourd,
    alors sa recréation à chaque rerun serait coûteuse et inutile.

    Ce cache permet donc :
    - d'améliorer fortement les performances,
    - d'éviter les temps de chargement répétés,
    - de stabiliser l'expérience utilisateur.

    Retour
    ------
    StyleRetriever
        Instance du moteur de recherche stylistique.
    """
    return StyleRetriever()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================
def _coerce_object_array(values) -> np.ndarray:
    """
    Convertit une collection en tableau NumPy de type object.

    But
    ---
    Uniformiser certains tableaux chargés depuis des fichiers .npy ou .npz,
    notamment ceux contenant :
    - des chaînes de caractères,
    - des chemins de fichiers,
    - des objets hétérogènes.

    Pourquoi dtype=object ?
    -----------------------
    Lorsque NumPy charge certaines structures, les types peuvent être :
    - implicites,
    - incohérents,
    - ou moins pratiques pour les comparaisons / affichages.

    En forçant dtype=object :
    - on garde une représentation souple,
    - on évite certaines conversions automatiques indésirables.

    Paramètres
    ----------
    values : any
        Données à convertir.

    Retour
    ------
    np.ndarray
        Tableau NumPy avec dtype=object.
    """
    return np.asarray(values, dtype=object)


def _build_style_names(labels: np.ndarray, classnames: np.ndarray) -> list[str]:
    """
    Transforme des labels numériques en noms de styles lisibles.

    But
    ---
    Associer à chaque label numérique un nom explicite de style pour
    l'affichage dans l'UMAP.

    Exemple
    -------
    labels      = [0, 2, 1]
    classnames  = ["Baroque", "Cubisme", "Impressionnisme"]

    Résultat :
    ["Baroque", "Impressionnisme", "Cubisme"]

    Cas limites gérés
    -----------------
    - label non convertible en entier
    - label hors bornes
    - structure classnames incohérente

    Dans ces cas, on renvoie un texte de secours :
        "Classe X"

    Paramètres
    ----------
    labels : np.ndarray
        Tableau des labels numériques.
    classnames : np.ndarray
        Tableau des noms de classes.

    Retour
    ------
    list[str]
        Liste des noms lisibles des styles.
    """
    style_names: list[str] = []

    for lab in labels:
        try:
            idx = int(lab)

            # Vérification de la validité de l'indice dans classnames
            if 0 <= idx < len(classnames):
                style_names.append(str(classnames[idx]))
            else:
                # Cas où le label pointe vers un indice inexistant
                style_names.append(f"Classe {lab}")

        except Exception:
            # Cas où le label ne peut pas être converti en entier
            style_names.append(f"Classe {lab}")

    return style_names


def _find_best_index(filenames: np.ndarray, best_filepath: str) -> int | None:
    """
    Retrouve l'indice du meilleur résultat dans la liste des fichiers UMAP.

    But
    ---
    Le moteur de retrieval renvoie un fichier top-1.
    Pour le mettre en évidence dans la visualisation UMAP, il faut retrouver
    à quel point du nuage ce fichier correspond.

    Stratégie
    ---------
    - convertir tous les chemins en chaînes
    - comparer à best_filepath
    - renvoyer l'indice du premier match trouvé

    Paramètres
    ----------
    filenames : np.ndarray
        Tableau des chemins de fichiers connus dans l'espace latent.
    best_filepath : str
        Chemin complet du meilleur résultat retourné.

    Retour
    ------
    int | None
        Indice du fichier s'il est trouvé, sinon None.

    Cas d'échec
    -----------
    Si aucun fichier ne correspond exactement, la fonction retourne None.
    """
    filenames_str = np.asarray([str(fp) for fp in filenames], dtype=object)

    matches = np.where(filenames_str == str(best_filepath))[0]

    if len(matches) == 0:
        return None

    return int(matches[0])


# =============================================================================
# CHARGEMENT DES DONNÉES LATENTES (UMAP + MÉTADONNÉES)
# =============================================================================
@st.cache_data
def load_latent_and_meta():
    """
    Charge les données nécessaires à la visualisation UMAP interactive.

    But
    ---
    Fournir à l'interface les données suivantes :
    - coordonnées UMAP 2D de chaque image
    - labels numériques
    - noms de classes
    - chemins des fichiers d'origine

    Sources prises en charge
    ------------------------
    1. Bundle unique :
       umap_bundle.npz

       contenant :
       - latent_2d
       - labels
       - classnames
       - filenames

    2. Fichiers séparés :
       - latent_2d.npy
       - labels.npy
       - classnames.npy
       - filenames.npy

    Pourquoi @st.cache_data ?
    -------------------------
    Ces données sont lues depuis le disque et ne changent pas à chaque
    interaction. Les mettre en cache :
    - évite les relectures répétées,
    - accélère le rendu,
    - diminue la charge inutile.

    Vérifications réalisées
    -----------------------
    - présence des clés nécessaires dans le bundle
    - validité de la forme du tableau latent_2d
    - cohérence du nombre de lignes entre :
      embeddings / labels / filenames

    Retour
    ------
    tuple | None
        Retourne (latent_2d, labels, classnames, filenames)
        ou None si les données UMAP ne sont pas disponibles.

    Exceptions
    ----------
    Peut lever une ValueError si les données sont incomplètes ou incohérentes.
    """
    cfg = load_config()

    # On lit dans la configuration le dossier racine des embeddings.
    emb_root = Path(cfg["paths"]["embeddings_root"])

    # Chemin d'un bundle UMAP "tout-en-un", s'il existe.
    bundle_path = emb_root / "umap_bundle.npz"

    # -------------------------------------------------------------------------
    # CAS 1 - Chargement depuis un bundle unique
    # -------------------------------------------------------------------------
    if bundle_path.exists():
        data = np.load(bundle_path, allow_pickle=True)

        # Clés indispensables pour pouvoir construire l'UMAP.
        required_keys = {"latent_2d", "labels", "classnames", "filenames"}

        # Recherche d'éventuelles clés manquantes.
        missing = required_keys.difference(set(data.files))

        if missing:
            raise ValueError(
                f"Le bundle UMAP est incomplet. Clés manquantes: {sorted(missing)}"
            )

        # Chargement explicite des différentes composantes
        latent_2d = np.asarray(data["latent_2d"])
        labels = np.asarray(data["labels"])
        classnames = _coerce_object_array(data["classnames"])
        filenames = _coerce_object_array(data["filenames"])

    # -------------------------------------------------------------------------
    # CAS 2 - Chargement depuis des fichiers séparés
    # -------------------------------------------------------------------------
    else:
        latent_path = emb_root / "latent_2d.npy"

        # Si la projection UMAP n'existe pas, on considère la visualisation
        # comme indisponible, sans faire planter toute l'application.
        if not latent_path.exists():
            return None

        latent_2d = np.load(latent_path)
        labels = np.load(emb_root / "labels.npy")
        classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)
        filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)

        # Normalisation des types chargés
        latent_2d = np.asarray(latent_2d)
        labels = np.asarray(labels)
        classnames = _coerce_object_array(classnames)
        filenames = _coerce_object_array(filenames)

    # -------------------------------------------------------------------------
    # VALIDATION DE LA STRUCTURE DE latent_2d
    # -------------------------------------------------------------------------
    # On attend ici une matrice de forme (N, 2) ou au moins (N, >=2),
    # car seules les deux premières dimensions sont exploitées pour l'affichage.
    if latent_2d.ndim != 2 or latent_2d.shape[1] < 2:
        raise ValueError(
            f"Projection UMAP invalide: shape={latent_2d.shape}, attendu=(N, 2)"
        )

    # -------------------------------------------------------------------------
    # VALIDATION DE LA COHÉRENCE ENTRE LES TABLEAUX
    # -------------------------------------------------------------------------
    n_latent = int(latent_2d.shape[0])
    n_labels = int(len(labels))
    n_filenames = int(len(filenames))

    if not (n_latent == n_labels == n_filenames):
        raise ValueError(
            "Incohérence entre latent_2d, labels et filenames.\n"
            f" - latent_2d : {n_latent}\n"
            f" - labels    : {n_labels}\n"
            f" - filenames : {n_filenames}\n"
            "Relance compute_embeddings.py puis visualization_umap.py."
        )

    return latent_2d, labels, classnames, filenames


# =============================================================================
# INITIALISATION DES RESSOURCES
# =============================================================================
# Chargement du moteur principal de recherche.
retriever = get_retriever()

# Variables de contrôle pour la partie UMAP.
# - latent_bundle : contiendra les données si elles existent
# - latent_error  : contiendra une éventuelle exception de chargement
latent_bundle = None
latent_error = None

# On essaie de charger les données UMAP dès l'initialisation.
# Si cela échoue, on capture l'erreur pour l'afficher plus tard à l'utilisateur
# sans interrompre le reste de l'application.
try:
    latent_bundle = load_latent_and_meta()
except Exception as exc:
    latent_error = exc


# =============================================================================
# INTERFACE UTILISATEUR - CONTRÔLES D'ENTRÉE
# =============================================================================
# Upload de l'image requête.
#
# type=[...]
# ----------
# Restreint les formats acceptés afin de :
# - guider l'utilisateur,
# - éviter certains cas non gérés,
# - limiter les erreurs en amont.
uploaded = st.file_uploader(
    "Upload une image (jpg/png/webp)",
    type=["jpg", "jpeg", "png", "webp"],
)

# Nombre de voisins similaires à retourner.
# Ici fixé à 4, conformément au besoin exprimé.
k = 4

# Activation ou non du calcul Grad-CAM.
# Cette option est laissée à l'utilisateur car elle peut être plus coûteuse.
show_gradcam = st.checkbox(
    "Afficher Grad-CAM (top-1)",
    value=False,
)


# =============================================================================
# TRAITEMENT PRINCIPAL
# =============================================================================
# Le cœur du traitement ne s'exécute que si l'utilisateur a chargé un fichier.
if uploaded is not None:
    # -------------------------------------------------------------------------
    # SAUVEGARDE TEMPORAIRE DE L'IMAGE UPLOADÉE
    # -------------------------------------------------------------------------
    # Pourquoi enregistrer un fichier temporaire ?
    # --------------------------------------------
    # Streamlit fournit un objet UploadedFile, qui est en mémoire.
    # Or le retriever attend vraisemblablement un chemin de fichier sur disque.
    #
    # On convertit donc le fichier uploadé en un fichier temporaire local.
    suffix = Path(uploaded.name).suffix if uploaded.name else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        query_path = f.name

    # -------------------------------------------------------------------------
    # RECHERCHE DES ŒUVRES LES PLUS SIMILAIRES
    # -------------------------------------------------------------------------
    # On encapsule l'appel dans un spinner pour indiquer à l'utilisateur
    # qu'un traitement est en cours.
    with st.spinner("Recherche des œuvres similaires..."):
        results = retriever.top_k_similar(query_path, k=k)

    # Sécurité : si aucun résultat n'est retourné, on affiche une erreur
    # claire et on stoppe l'exécution de ce rerun.
    if not results:
        st.error("Aucun résultat n'a été retourné par le moteur de recherche.")
        st.stop()

    # Le premier résultat est considéré comme le meilleur match.
    best = results[0]

    # -------------------------------------------------------------------------
    # AFFICHAGE DE L'IMAGE REQUÊTE
    # -------------------------------------------------------------------------
    st.subheader("Image source")
    st.image(
        query_path,
        caption="Image requête",
        width="stretch",
    )

    # -------------------------------------------------------------------------
    # AFFICHAGE DES MEILLEURES CORRESPONDANCES
    # -------------------------------------------------------------------------
    st.subheader("Comparaison visuelle")

    # On crée jusqu'à 4 colonnes, ou moins si moins de résultats sont présents.
    cols = st.columns(min(4, len(results)))

    for i, res in enumerate(results):
        with cols[i % len(cols)]:
            st.image(
                res["filepath"],
                caption=f"Top {i + 1} — {res['style']} ({res['similarity']:.3f})",
                width="stretch",
            )

    # Affichage textuel du style proposé à partir du meilleur match.
    st.markdown(f"**Style suggéré :** {best['style']}")

    # -------------------------------------------------------------------------
    # VERSION ALTERNATIVE D'AFFICHAGE (CONSERVÉE EN COMMENTAIRE)
    # -------------------------------------------------------------------------
    # Cette section est volontairement laissée commentée.
    # Elle constitue une variante d'affichage où l'image requête et les top-k
    # résultats sont tous placés sur une seule ligne.
    #
    # Ce type de bloc commenté peut être utile :
    # - pour expérimenter d'autres layouts,
    # - pour revenir rapidement à un ancien affichage,
    # - pour comparer deux stratégies UI.
    #
    # st.subheader("Comparaison visuelle")
    #
    # cols = st.columns(len(results) + 1)
    #
    # with cols[0]:
    #     st.image(
    #         query_path,
    #         caption="Image requête",
    #         width="stretch",
    #     )
    #
    # for i, res in enumerate(results, start=1):
    #     with cols[i]:
    #         st.image(
    #             res["filepath"],
    #             caption=f"Top {i} — {res['style']} ({res['similarity']:.3f})",
    #             width="stretch",
    #         )
    #
    # st.markdown(f"**Style suggéré :** {best['style']}")

    # -------------------------------------------------------------------------
    # TABLEAU RÉCAPITULATIF
    # -------------------------------------------------------------------------
    st.subheader("Résumé des résultats")

    # Construction d'un DataFrame de synthèse.
    #
    # Colonnes :
    # - rang        : position dans le classement
    # - style       : style associé au résultat
    # - similarité  : score numérique
    # - fichier     : nom de fichier court
    # - chemin      : chemin complet
    #
    # Pourquoi conserver aussi le chemin complet ?
    # --------------------------------------------
    # Cela peut être utile pour :
    # - le débogage,
    # - la traçabilité,
    # - vérifier l'origine exacte d'un résultat.
    df_results = pd.DataFrame(
        [
            {
                "rang": i + 1,
                "style": res["style"],
                "similarité": round(float(res["similarity"]), 4),
                "fichier": Path(str(res["filepath"])).name,
                "chemin": str(res["filepath"]),
            }
            for i, res in enumerate(results)
        ]
    )

    st.dataframe(df_results, width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # EXPLICATION VISUELLE AVEC GRAD-CAM
    # -------------------------------------------------------------------------
    if show_gradcam:
        st.subheader("Explication visuelle (Grad-CAM similarity)")

        try:
            # L'explication porte ici sur la relation entre :
            # - l'image requête
            # - le meilleur résultat (top-1)
            with st.spinner("Calcul des cartes Grad-CAM..."):
                explanation = retriever.explain_similarity(
                    query_path,
                    best["filepath"],
                )

            # Deux colonnes pour comparer visuellement les deux overlays.
            c1, c2 = st.columns(2)

            with c1:
                st.image(
                    explanation["query_overlay"],
                    caption=f"Requête (couche: {explanation['target_layer']})",
                    width="stretch",
                )

            with c2:
                st.image(
                    explanation["candidate_overlay"],
                    caption=f"Top-1 match ({best['style']})",
                    width="stretch",
                )

            # Rappel du score de similarité numérique.
            st.caption(f"Similarité (cosine): {explanation['similarity']:.3f}")

        except Exception as exc:
            # L'application reste robuste si l'explication échoue.
            # On préfère avertir plutôt que faire planter toute l'interface.
            st.warning(f"Grad-CAM indisponible : {exc}")

    else:
        # Message d'accompagnement lorsque l'option n'est pas active.
        st.info(
            "Active l'option Grad-CAM pour visualiser les zones qui "
            "contribuent à la similarité du top-1."
        )

    # -------------------------------------------------------------------------
    # VISUALISATION DE L'ESPACE LATENT AVEC UMAP
    # -------------------------------------------------------------------------
    # Cas 1 : une erreur a été rencontrée lors du chargement des données UMAP
    if latent_error is not None:
        st.warning(f"Visualisation UMAP indisponible : {latent_error}")

    # Cas 2 : les données UMAP sont bien disponibles
    elif latent_bundle is not None:
        st.subheader("Espace latent (UMAP interactif)")

        latent_2d, labels, classnames, filenames = latent_bundle

        # Construction des noms de styles lisibles.
        style_names = _build_style_names(labels, classnames)

        # Extraction du nom court des fichiers pour affichage plus propre.
        short_filenames = [Path(str(fp)).name for fp in filenames]

        # DataFrame principal utilisé par Plotly.
        df_umap = pd.DataFrame(
            {
                "x": latent_2d[:, 0],
                "y": latent_2d[:, 1],
                "label": labels,
                "style": style_names,
                "filename": short_filenames,
                "filepath": [str(fp) for fp in filenames],
            }
        )

        # Liste triée des styles disponibles dans les données.
        styles_disponibles = sorted(df_umap["style"].astype(str).unique().tolist())

        # Multisélection permettant à l'utilisateur de filtrer les classes visibles.
        styles_selectionnes = st.multiselect(
            "Filtrer les styles affichés dans l'UMAP",
            options=styles_disponibles,
            default=styles_disponibles,
        )

        # On applique le filtre choisi par l'utilisateur.
        df_umap_filtered = df_umap[df_umap["style"].isin(styles_selectionnes)].copy()

        if df_umap_filtered.empty:
            st.warning("Aucun point à afficher : aucun style sélectionné.")
        else:
            # Création du scatter plot interactif.
            fig = px.scatter(
                df_umap_filtered,
                x="x",
                y="y",
                color="style",
                hover_data={
                    # On n'affiche pas x/y dans le tooltip pour éviter le bruit,
                    # sauf si cela devient utile plus tard pour du debug.
                    "x": False,
                    "y": False,
                    "label": True,
                    "style": True,
                    "filename": True,
                    "filepath": False,
                },
                opacity=0.45,
                title="Projection UMAP des embeddings",
            )

            # Uniformisation de la taille des points standards.
            fig.update_traces(marker=dict(size=7))

            # -----------------------------------------------------------------
            # SURBRILLANCE DU TOP-1 DANS L'ESPACE LATENT
            # -----------------------------------------------------------------
            idx_best = _find_best_index(filenames, str(best["filepath"]))

            if idx_best is not None:
                x_best = latent_2d[idx_best, 0]
                y_best = latent_2d[idx_best, 1]
                best_filename = Path(str(best["filepath"])).name

                # Ajout d'une trace dédiée au meilleur résultat.
                # On le rend visuellement plus saillant :
                # - taille plus grande
                # - cercle ouvert
                # - contour noir
                # - texte au-dessus
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
                            f"Fichier: {best_filename}<br>"
                            f"Similarité: {best['similarity']:.3f}"
                            "<extra></extra>"
                        ),
                        showlegend=True,
                    )
                )

                # Ajout d'une annotation visible directement sur la figure.
                fig.add_annotation(
                    x=x_best,
                    y=y_best,
                    text=f"Top-1 : {best['style']}",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )

            # Réglages finaux de mise en page.
            fig.update_layout(
                xaxis_title="Dimension UMAP 1",
                yaxis_title="Dimension UMAP 2",
                legend_title="Styles",
                height=700,
            )

            st.plotly_chart(fig, width="stretch")

            # Texte d'aide sous la figure.
            st.caption(
                "Chaque point représente une œuvre projetée dans un espace latent 2D. "
                "Les couleurs correspondent aux styles artistiques. "
                "Le point entouré correspond au meilleur résultat (top-1)."
            )

# =============================================================================
# CAS PAR DÉFAUT : AUCUNE IMAGE ENTRÉE
# =============================================================================
else:
    # Message d'accueil / guidance tant qu'aucune image n'a été chargée.
    st.info("Charge une image pour lancer la recherche.")
