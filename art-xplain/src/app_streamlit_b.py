


"""
Application Streamlit pour rechercher des œuvres similaires visuellement
à partir d'une image envoyée par l'utilisateur.

Fonctionnalités :
- Upload d'une image
- Recherche des 4 œuvres les plus similaires
- Affichage de l'image source au-dessus des 4 images de comparaison
- Affichage optionnel d'une explication visuelle avec Grad-CAM
- Visualisation optionnelle dans un espace latent 2D (UMAP) avec Plotly
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------
# Gestion des imports du projet
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.retrieval import StyleRetriever


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
    return StyleRetriever()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _coerce_object_array(values) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _build_style_names(labels: np.ndarray, classnames: np.ndarray) -> list[str]:
    style_names: list[str] = []
    for lab in labels:
        try:
            idx = int(lab)
            if 0 <= idx < len(classnames):
                style_names.append(str(classnames[idx]))
            else:
                style_names.append(f"Classe {lab}")
        except Exception:
            style_names.append(f"Classe {lab}")
    return style_names


def _find_best_index(filenames: np.ndarray, best_filepath: str) -> int | None:
    filenames_str = np.asarray([str(fp) for fp in filenames], dtype=object)
    matches = np.where(filenames_str == str(best_filepath))[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


# -------------------------------------------------------------------
# Chargement UMAP + métadonnées
# -------------------------------------------------------------------
@st.cache_data
def load_latent_and_meta():
    cfg = load_config()
    emb_root = Path(cfg["paths"]["embeddings_root"])

    bundle_path = emb_root / "umap_bundle.npz"
    if bundle_path.exists():
        data = np.load(bundle_path, allow_pickle=True)

        required_keys = {"latent_2d", "labels", "classnames", "filenames"}
        missing = required_keys.difference(set(data.files))
        if missing:
            raise ValueError(
                f"Le bundle UMAP est incomplet. Clés manquantes: {sorted(missing)}"
            )

        latent_2d = np.asarray(data["latent_2d"])
        labels = np.asarray(data["labels"])
        classnames = _coerce_object_array(data["classnames"])
        filenames = _coerce_object_array(data["filenames"])
    else:
        latent_path = emb_root / "latent_2d.npy"
        if not latent_path.exists():
            return None

        latent_2d = np.load(latent_path)
        labels = np.load(emb_root / "labels.npy")
        classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)
        filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)

        latent_2d = np.asarray(latent_2d)
        labels = np.asarray(labels)
        classnames = _coerce_object_array(classnames)
        filenames = _coerce_object_array(filenames)

    if latent_2d.ndim != 2 or latent_2d.shape[1] < 2:
        raise ValueError(
            f"Projection UMAP invalide: shape={latent_2d.shape}, attendu=(N, 2)"
        )

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


# -------------------------------------------------------------------
# Initialisation des ressources
# -------------------------------------------------------------------
retriever = get_retriever()

latent_bundle = None
latent_error = None
try:
    latent_bundle = load_latent_and_meta()
except Exception as exc:
    latent_error = exc


# -------------------------------------------------------------------
# Interface utilisateur
# -------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload une image (jpg/png/webp)",
    type=["jpg", "jpeg", "png", "webp"],
)

k = 4

show_gradcam = st.checkbox(
    "Afficher Grad-CAM (top-1)",
    value=False,
)


# -------------------------------------------------------------------
# Traitement principal
# -------------------------------------------------------------------
if uploaded is not None:
    suffix = Path(uploaded.name).suffix if uploaded.name else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        query_path = f.name

    # ---------------------------------------------------------------
    # Recherche des œuvres similaires
    # ---------------------------------------------------------------
    with st.spinner("Recherche des œuvres similaires..."):
        results = retriever.top_k_similar(query_path, k=k)

    if not results:
        st.error("Aucun résultat n'a été retourné par le moteur de recherche.")
        st.stop()

    best = results[0]

    # ---------------------------------------------------------------
    # Affichage : image source au-dessus des images de comparaison
    # ---------------------------------------------------------------
    st.subheader("Image source")
    st.image(
        query_path,
        caption="Image requête",
        width="stretch",
    )

    st.subheader("Comparaison visuelle")
    cols = st.columns(min(4, len(results)))

    for i, res in enumerate(results):
        with cols[i % len(cols)]:
            st.image(
                res["filepath"],
                caption=f"Top {i + 1} — {res['style']} ({res['similarity']:.3f})",
                width="stretch",
            )

    st.markdown(f"**Style suggéré :** {best['style']}")

    # ---------------------------------------------------------------
    # Tableau récapitulatif
    # ---------------------------------------------------------------
    st.subheader("Résumé des résultats")
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

    # ---------------------------------------------------------------
    # Explication visuelle Grad-CAM
    # ---------------------------------------------------------------
    if show_gradcam:
        st.subheader("Explication visuelle (Grad-CAM similarity)")

        try:
            with st.spinner("Calcul des cartes Grad-CAM..."):
                explanation = retriever.explain_similarity(
                    query_path,
                    best["filepath"],
                )

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

            st.caption(f"Similarité (cosine): {explanation['similarity']:.3f}")

        except Exception as exc:
            st.warning(f"Grad-CAM indisponible : {exc}")
    else:
        st.info(
            "Active l'option Grad-CAM pour visualiser les zones qui "
            "contribuent à la similarité du top-1."
        )

    # ---------------------------------------------------------------
    # UMAP interactif avec Plotly
    # ---------------------------------------------------------------
    if latent_error is not None:
        st.warning(f"Visualisation UMAP indisponible : {latent_error}")

    elif latent_bundle is not None:
        st.subheader("Espace latent (UMAP interactif)")

        latent_2d, labels, classnames, filenames = latent_bundle

        style_names = _build_style_names(labels, classnames)
        short_filenames = [Path(str(fp)).name for fp in filenames]

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

        styles_disponibles = sorted(df_umap["style"].astype(str).unique().tolist())
        styles_selectionnes = st.multiselect(
            "Filtrer les styles affichés dans l'UMAP",
            options=styles_disponibles,
            default=styles_disponibles,
        )

        df_umap_filtered = df_umap[df_umap["style"].isin(styles_selectionnes)].copy()

        if df_umap_filtered.empty:
            st.warning("Aucun point à afficher : aucun style sélectionné.")
        else:
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
                title="Projection UMAP des embeddings",
            )

            fig.update_traces(marker=dict(size=7))

            idx_best = _find_best_index(filenames, str(best["filepath"]))
            if idx_best is not None:
                x_best = latent_2d[idx_best, 0]
                y_best = latent_2d[idx_best, 1]
                best_filename = Path(str(best["filepath"])).name

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

            fig.update_layout(
                xaxis_title="Dimension UMAP 1",
                yaxis_title="Dimension UMAP 2",
                legend_title="Styles",
                height=700,
            )

            st.plotly_chart(fig, width="stretch")

            st.caption(
                "Chaque point représente une œuvre projetée dans un espace latent 2D. "
                "Les couleurs correspondent aux styles artistiques. "
                "Le point entouré correspond au meilleur résultat (top-1)."
            )

else:
    st.info("Charge une image pour lancer la recherche.")
