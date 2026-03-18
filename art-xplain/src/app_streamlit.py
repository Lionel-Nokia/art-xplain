"""
Application Streamlit pour rechercher des œuvres similaires visuellement
à partir d'une image envoyée par l'utilisateur.

Fonctionnalités :
- Upload d'une image
- Recherche des k œuvres les plus similaires
- Affichage des résultats
- Affichage optionnel d'une explication visuelle avec Grad-CAM
- Visualisation optionnelle dans un espace latent 2D (UMAP) avec Plotly

Cette version est commentée et inclut :
- un UMAP interactif,
- une légende des styles,
- un survol informatif,
- une mise en évidence du meilleur résultat.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------
# Gestion des imports du projet
# -------------------------------------------------------------------
# On ajoute la racine du projet au PYTHONPATH pour permettre l'import
# des modules locaux, même si l'application est lancée via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import StyleRetriever
from src.utils import load_config


# -------------------------------------------------------------------
# Configuration générale de la page
# -------------------------------------------------------------------
st.set_page_config(page_title="Art-Xplain", layout="wide")
st.title("Art-Xplain — Similarité stylistique")


# -------------------------------------------------------------------
# Chargement du moteur de recherche
# -------------------------------------------------------------------
@st.cache_resource
def get_retriever() -> StyleRetriever:
    """
    Crée et met en cache l'objet StyleRetriever.

    Le cache évite de recharger le modèle ou l'index à chaque interaction
    de l'utilisateur (slider, checkbox, etc.).
    """
    return StyleRetriever()


# -------------------------------------------------------------------
# Chargement des données UMAP et métadonnées associées
# -------------------------------------------------------------------
@st.cache_data
def load_latent_and_meta():
    """
    Charge la projection UMAP 2D et les métadonnées associées.

    Fichiers attendus dans embeddings_root :
    - latent_2d.npy
    - labels.npy
    - classnames.npy
    - filenames.npy

    Retourne :
        (latent_2d, labels, classnames, filenames)
    ou :
        None si les données UMAP ne sont pas disponibles.
    """
    cfg = load_config()
    emb_root = Path(cfg["paths"]["embeddings_root"])

    latent_path = emb_root / "latent_2d.npy"
    if not latent_path.exists():
        return None

    latent_2d = np.load(latent_path)
    labels = np.load(emb_root / "labels.npy")
    classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)
    filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)

    return latent_2d, labels, classnames, filenames


# -------------------------------------------------------------------
# Initialisation des ressources
# -------------------------------------------------------------------
retriever = get_retriever()
latent_bundle = load_latent_and_meta()


# -------------------------------------------------------------------
# Interface utilisateur
# -------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload une image (jpg/png)",
    type=["jpg", "jpeg", "png", "webp"]
)

# k = st.slider(
#     "Top-k similaires",
#     min_value=1,
#     max_value=10,
#     value=3
# )

# st.write("k =", k)
k = 3

show_gradcam = st.checkbox(
    "Afficher Grad-CAM (top-1)",
    value=False
)


# -------------------------------------------------------------------
# Traitement principal
# -------------------------------------------------------------------
if uploaded is not None:
    # ---------------------------------------------------------------
    # Sauvegarde temporaire du fichier uploadé
    # ---------------------------------------------------------------
    # Le retriever attend un chemin vers un fichier sur disque.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(uploaded.read())
        query_path = f.name

    # ---------------------------------------------------------------
    # Affichage de l'image requête
    # ---------------------------------------------------------------
    st.image(
        query_path,
        caption="Image requête",
        width="stretch"
    )

    # ---------------------------------------------------------------
    # Recherche des œuvres similaires
    # ---------------------------------------------------------------
    with st.spinner("Recherche des œuvres similaires..."):
        results = retriever.top_k_similar(query_path, k=int(k))

    # Sécurité minimale si aucun résultat n'est retourné
    if not results:
        st.error("Aucun résultat n'a été retourné par le moteur de recherche.")
        st.stop()

    # ---------------------------------------------------------------
    # Affichage des résultats
    # ---------------------------------------------------------------
    st.subheader("Résultats (Top-k)")

    cols = st.columns(min(4, len(results)))
    for i, res in enumerate(results):
        col = cols[i % len(cols)]
        col.image(
            res["filepath"],
            caption=f"{res['style']} (sim={res['similarity']:.3f})",
            width="stretch"
        )

    # ---------------------------------------------------------------
    # Style suggéré = top-1
    # ---------------------------------------------------------------
    best = results[0]
    st.markdown(f"**Style suggéré :** {best['style']}")

    # ---------------------------------------------------------------
    # Explication visuelle Grad-CAM
    # ---------------------------------------------------------------
    if show_gradcam:
        st.subheader("Explication visuelle (Grad-CAM similarity)")

        with st.spinner("Calcul des cartes Grad-CAM..."):
            explanation = retriever.explain_similarity(
                query_path,
                best["filepath"]
            )

        c1, c2 = st.columns(2)

        c1.image(
            explanation["query_overlay"],
            caption=f"Requête (couche: {explanation['target_layer']})",
            use_container_width=True,
        )

        c2.image(
            explanation["candidate_overlay"],
            caption=f"Top-1 match ({best['style']})",
            use_container_width=True,
        )

        st.caption(f"Similarité (cosine): {explanation['similarity']:.3f}")

    # ---------------------------------------------------------------
    # UMAP interactif avec Plotly
    # ---------------------------------------------------------------
    if latent_bundle is not None:
        st.subheader("Espace latent (UMAP interactif)")

        latent_2d, labels, classnames, filenames = latent_bundle

        # -----------------------------------------------------------
        # Préparation des données
        # -----------------------------------------------------------
        # Conversion des labels numériques en noms de styles
        style_names = []
        for lab in labels:
            try:
                style_names.append(str(classnames[int(lab)]))
            except Exception:
                style_names.append(f"Classe {lab}")

        # Nom court des fichiers pour affichage au survol
        short_filenames = [Path(str(fp)).name for fp in filenames]

        df_umap = pd.DataFrame({
            "x": latent_2d[:, 0],
            "y": latent_2d[:, 1],
            "label": labels,
            "style": style_names,
            "filename": short_filenames,
            "filepath": filenames,
        })

        # -----------------------------------------------------------
        # Filtre optionnel par style
        # -----------------------------------------------------------
        styles_disponibles = sorted(df_umap["style"].astype(str).unique().tolist())
        styles_selectionnes = st.multiselect(
            "Filtrer les styles affichés dans l'UMAP",
            options=styles_disponibles,
            default=styles_disponibles
        )

        df_umap_filtered = df_umap[df_umap["style"].isin(styles_selectionnes)].copy()

        if df_umap_filtered.empty:
            st.warning("Aucun point à afficher : aucun style sélectionné.")
        else:
            # -------------------------------------------------------
            # Scatter interactif principal
            # -------------------------------------------------------
            fig = px.scatter(
                df_umap_filtered,
                x="x",
                y="y",
                color="style",
                hover_data={
                    "x": False,
                    "y": False,
                    "label": True,
                    "style": True,
                    "filename": True,
                    "filepath": False,
                },
                opacity=0.45,
                title="Projection UMAP des embeddings"
            )

            fig.update_traces(marker=dict(size=7))

            # -------------------------------------------------------
            # Mise en évidence du meilleur résultat (top-1)
            # -------------------------------------------------------
            try:
                idx_best = int(np.where(filenames == best["filepath"])[0][0])

                x_best = latent_2d[idx_best, 0]
                y_best = latent_2d[idx_best, 1]
                best_filename = Path(str(best["filepath"])).name

                # Ajout d'un point spécial pour le top-1
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
                            line=dict(width=3, color="black")
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

                # Annotation supplémentaire plus visible
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

            except Exception:
                st.warning("Impossible de localiser le meilleur résultat dans les métadonnées UMAP.")

            # -------------------------------------------------------
            # Mise en forme du graphique
            # -------------------------------------------------------
            fig.update_layout(
                xaxis_title="Dimension UMAP 1",
                yaxis_title="Dimension UMAP 2",
                legend_title="Styles",
                height=700,
            )

            st.plotly_chart(fig, use_container_width=True)

            # -------------------------------------------------------
            # Légende explicative textuelle
            # -------------------------------------------------------
            st.caption(
                "Chaque point représente une œuvre projetée dans un espace latent 2D. "
                "Les couleurs correspondent aux styles artistiques. "
                "Le point entouré en vert correspond au meilleur résultat (top-1). "
                "Tu peux zoomer, survoler les points et filtrer les styles affichés."
            )

    # ---------------------------------------------------------------
    # Message d'aide si Grad-CAM n'est pas activé
    # ---------------------------------------------------------------
    if not show_gradcam:
        st.info(
            "Active l'option Grad-CAM pour visualiser les zones qui "
            "contribuent à la similarité du top-1."
        )
