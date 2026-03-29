from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.utils import load_config, resolve_project_path, resolve_stored_path


def load_inline_svg(svg_path: Path) -> str:
    # On injecte le SVG inline pour pouvoir l'habiller via le CSS Streamlit.
    # C'est plus souple qu'un simple `st.image` lorsqu'on veut l'intégrer
    # dans un header HTML custom.
    try:
        return svg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def compute_source_image_display_width(image_path: str) -> int:
    # Le but n'est pas de reproduire la taille réelle de l'image,
    # mais d'éviter qu'une œuvre très verticale ou panoramique casse
    # l'équilibre visuel de la section "image source".
    fallback_width = 600
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return fallback_width
    if width <= 0 or height <= 0:
        return fallback_width
    ratio = width / height
    if ratio >= 1.6:
        target_width = 760
    elif ratio >= 1.15:
        target_width = 680
    elif ratio >= 0.85:
        target_width = 600
    elif ratio >= 0.6:
        target_width = 480
    else:
        target_width = 380
    return max(320, min(target_width, width))


def setup_page(logo_path: Path) -> None:
    # Toute la direction artistique Streamlit est centralisée ici.
    # Pour la revue d'équipe, c'est le bon endroit si l'on veut :
    # - faire évoluer le thème global,
    # - harmoniser les composants de l'app,
    # - ou isoler plus tard cette feuille de style dans un asset dédié.
    st.set_page_config(page_title="Art-Xplain", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --museum-paper: #f6f3ee;
            --museum-panel: rgba(255, 255, 255, 0.94);
            --museum-panel-strong: rgba(255, 255, 255, 0.985);
            --museum-ink: #111111;
            --museum-muted: #575757;
            --museum-line: rgba(17, 17, 17, 0.14);
            --museum-accent: #b1221c;
            --museum-accent-soft: rgba(177, 34, 28, 0.08);
            --museum-button-bg: rgba(255, 255, 255, 0.96);
            --museum-button-bg-hover: #f6f3ee;
            --museum-shadow: 0 14px 34px rgba(17, 17, 17, 0.06);
            --museum-font: "Georgia", "Iowan Old Style", "Times New Roman", serif;
        }
        html, body, [class*="css"] { font-family: var(--museum-font); color-scheme: light; }
        .stApp { color: var(--museum-ink); background: #ffffff; }
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.88); backdrop-filter: blur(8px); border-bottom: 1px solid rgba(17,17,17,0.08); }
        [data-testid="block-container"] { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1200px; }
        h1, h2, h3 { font-family: var(--museum-font); color: var(--museum-ink); letter-spacing: 0.01em; }
        h3 { font-size: 1.6rem; margin-top: 1.4rem; }
        p, li, label, .stCaption { color: var(--museum-muted); }
        .museum-hero { padding: 2.2rem 2rem 1.9rem 2rem; margin-bottom: 1.5rem; border: 1px solid var(--museum-line); border-radius: 10px; background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(246,243,238,0.78)), linear-gradient(90deg, rgba(255,255,255,0.38), rgba(255,255,255,0.12)), url("https://www.mam.paris.fr/sites/default/files/home/bandeau/hall_0.png"); background-size: cover; background-position: center center; box-shadow: var(--museum-shadow); position: relative; overflow: hidden; }
        .museum-hero-header { display:flex; align-items:flex-start; gap:1.3rem; }
        .museum-logo-wrap { width:176px; min-width:176px; padding-top:0.2rem; }
        .museum-logo-wrap svg { display:block; width:100%; height:auto; }
        .museum-hero-copy { flex:1; min-width:0; }
        .museum-hero::after { content:""; position:absolute; left:2rem; right:2rem; top:0.95rem; height:4px; background:var(--museum-accent); }
        .museum-kicker { display:inline-block; margin-bottom:0.9rem; color:var(--museum-accent); font-size:0.74rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; }
        .museum-title { margin:0.15rem 0 0 0; font-size:clamp(2.6rem,4vw,4.9rem); line-height:0.96; }
        .museum-lead { max-width:52rem; margin:1rem 0 0 0; font-size:1rem; line-height:1.72; color:var(--museum-muted); }
        .museum-meta { margin-top:1.4rem; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--museum-muted); }
        .museum-card { padding:1rem 1.1rem; margin:0.45rem 0 0.8rem 0; border:1px solid var(--museum-line); border-radius:8px; background:var(--museum-panel); box-shadow:var(--museum-shadow); }
        .museum-card strong { display:block; color:var(--museum-ink); font-size:1.08rem; margin-bottom:0.15rem; }
        .museum-card em { color:var(--museum-muted); font-size:0.96rem; }
        .museum-style-chip { display:inline-block; margin:0.15rem 0 1rem 0; padding:0.48rem 0.88rem; border-radius:4px; border:1px solid rgba(177,34,28,0.22); background:var(--museum-accent-soft); color:var(--museum-accent); font-size:0.82rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; }
        [data-testid="stFileUploaderDropzone"], [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stAlert"] { border:1px solid var(--museum-line)!important; border-radius:8px!important; background:var(--museum-panel)!important; box-shadow:var(--museum-shadow); }
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] small {
            font-family: Arial, Helvetica, sans-serif !important;
            color: var(--museum-ink) !important;
            -webkit-text-fill-color: currentColor !important;
        }
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stFileUploaderDropzone"] p {
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            line-height: 1.35 !important;
            letter-spacing: 0 !important;
            margin: 0 !important;
        }
        [data-testid="stFileUploaderDropzone"] small {
            font-size: 0.95rem !important;
            font-weight: 400 !important;
            color: var(--museum-muted) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            color: transparent !important;
            -webkit-text-fill-color: transparent !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            letter-spacing: 0 !important;
            position: relative !important;
            min-width: 15rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
        [data-testid="stFileUploaderDropzone"] button::after {
            content: "Upload";
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 1rem;
            font-weight: 500;
            color: var(--museum-ink);
            -webkit-text-fill-color: var(--museum-ink);
            white-space: nowrap;
            pointer-events: none;
        }
        [data-testid="stImage"] img { border-radius:4px; border:1px solid var(--museum-line); box-shadow:0 16px 32px rgba(16,16,16,0.10); }
        [data-testid="stImage"] img, [data-testid="stImage"] picture, [data-testid="stImage"] > div { max-width:100%; }
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { overflow-x:auto; }
        .stMarkdown a { color: var(--museum-accent); }
        button,
        [role="button"],
        .stButton > button,
        .stDownloadButton > button,
        .stForm [data-testid="stFormSubmitButton"] > button {
            -webkit-appearance: none !important;
            appearance: none !important;
            background: var(--museum-button-bg) !important;
            background-color: var(--museum-button-bg) !important;
            background-image: none !important;
            color: var(--museum-ink) !important;
            -webkit-text-fill-color: var(--museum-ink) !important;
            border: 1px solid var(--museum-line) !important;
            border-radius: 999px !important;
            box-shadow: none !important;
            color-scheme: light !important;
            opacity: 1 !important;
        }
        button:hover,
        [role="button"]:hover,
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stForm [data-testid="stFormSubmitButton"] > button:hover {
            background: var(--museum-button-bg-hover) !important;
            background-color: var(--museum-button-bg-hover) !important;
            border-color: rgba(17, 17, 17, 0.24) !important;
        }
        button:focus,
        [role="button"]:focus,
        .stButton > button:focus,
        .stDownloadButton > button:focus,
        .stForm [data-testid="stFormSubmitButton"] > button:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(177, 34, 28, 0.14) !important;
        }
        [data-testid="stCheckbox"] {
            background: transparent !important;
            color-scheme: light !important;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] div {
            background: transparent !important;
            color: var(--museum-ink) !important;
            -webkit-text-fill-color: var(--museum-ink) !important;
        }
        [data-testid="stCheckbox"] label > div:first-child,
        [data-testid="stCheckbox"] label > div:first-child > div,
        [data-testid="stCheckbox"] [role="checkbox"],
        [data-testid="stCheckbox"] [data-checked] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            background-image: none !important;
            border-color: rgba(17, 17, 17, 0.28) !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.55) !important;
            color-scheme: light !important;
        }
        [data-testid="stCheckbox"] input[type="checkbox"] {
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 1.05rem !important;
            height: 1.05rem !important;
            margin: 0 !important;
            border: 1px solid rgba(17, 17, 17, 0.28) !important;
            border-radius: 0.28rem !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            background-image: none !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.55) !important;
            color-scheme: light !important;
            position: relative !important;
            vertical-align: middle !important;
        }
        [data-testid="stCheckbox"] input[type="checkbox"] + div,
        [data-testid="stCheckbox"] input[type="checkbox"] ~ div {
            background: #ffffff !important;
            background-color: #ffffff !important;
            background-image: none !important;
            border-color: rgba(17, 17, 17, 0.28) !important;
        }
        [data-testid="stCheckbox"] input[type="checkbox"]:checked {
            background: var(--museum-accent) !important;
            background-color: var(--museum-accent) !important;
            border-color: var(--museum-accent) !important;
        }
        [data-testid="stCheckbox"] label[data-checked="true"] > div:first-child,
        [data-testid="stCheckbox"] label[data-checked="true"] > div:first-child > div,
        [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
            background: var(--museum-accent) !important;
            background-color: var(--museum-accent) !important;
            border-color: var(--museum-accent) !important;
        }
        [data-testid="stCheckbox"] input[type="checkbox"]:checked::after {
            content: "" !important;
            position: absolute !important;
            left: 0.33rem !important;
            top: 0.08rem !important;
            width: 0.24rem !important;
            height: 0.5rem !important;
            border: solid #ffffff !important;
            border-width: 0 0.12rem 0.12rem 0 !important;
            transform: rotate(45deg) !important;
        }
        [data-testid="stCheckbox"] input[type="checkbox"]:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(177, 34, 28, 0.14) !important;
        }
        @media (max-width: 900px) {
            [data-testid="block-container"] { padding-top:1.25rem; padding-bottom:2.5rem; padding-left:1rem; padding-right:1rem; }
            .museum-hero { padding:1.35rem 1rem 1.15rem 1rem; border-radius:8px; background-position:62% center; }
            .museum-hero::after { left:1rem; right:1rem; top:0.7rem; }
            .museum-title { font-size:clamp(2rem,10vw,3.1rem); line-height:1; }
            .museum-kicker, .museum-meta, .museum-style-chip { letter-spacing:0.08em; }
            .museum-lead { font-size:0.96rem; line-height:1.55; }
            [data-testid="stHorizontalBlock"] { flex-wrap:wrap!important; flex-direction:column!important; align-items:stretch!important; gap:0.85rem; }
            [data-testid="column"] { width:100%!important; flex:1 1 100%!important; min-width:0!important; }
            [data-testid="stImage"], [data-testid="stImage"] > div, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { width:100%!important; max-width:100%!important; min-width:0!important; }
            .js-plotly-plot, .plot-container, .svg-container { width:100%!important; max-width:100%!important; }
            [data-testid="stImage"] img { width:100%!important; height:auto!important; max-width:100%!important; }
            .museum-card { padding:0.85rem 0.9rem; }
            [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { overflow-wrap:anywhere; }
            [data-testid="stFileUploaderDropzone"] {
                padding: 0.9rem !important;
            }
            [data-testid="stFileUploaderDropzone"] section {
                flex-direction: column !important;
                align-items: stretch !important;
                gap: 0.75rem !important;
            }
            [data-testid="stFileUploaderDropzone"] button {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                margin-left: 0 !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            [data-testid="stFileUploaderDropzone"] button::after {
                font-size: 0.95rem !important;
            }
            [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stFileUploaderDropzone"] p {
                font-size: 1rem !important;
                line-height: 1.3 !important;
            }
            [data-testid="stFileUploaderDropzone"] small {
                font-size: 0.9rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <section class="museum-hero">
            <div class="museum-hero-header">
                <div class="museum-logo-wrap">{load_inline_svg(logo_path)}</div>
                <div class="museum-hero-copy">
                    <span class="museum-kicker">Musee d'Art Numérique & Datascience</span>
                    <h1 class="museum-title">Art-Xplain</h1>
                    <p class="museum-lead">
                        Exploration des proximités stylistiques entre œuvres, à partir d'une image source,
                        de voisinages visuels et d'une lecture analytique par embeddings, UMAP et Grad-CAM++.
                    </p>
                    <div class="museum-meta">Collections numeriques • Similarite stylistique • Analyse visuelle</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def inspect_runtime_assets() -> dict[str, object]:
    # Cette vérification agit comme un garde-fou applicatif :
    # avant d'autoriser l'upload, on confirme que le modèle,
    # les embeddings et les données images sont cohérents.
    # Cela évite beaucoup d'erreurs tardives plus difficiles à comprendre côté UI.
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
    model_detail = f"encodeur trouvé: {model_path.name}" if model_ok else f"fichier manquant: {model_path.name}"
    required_embedding_files = [emb_root / name for name in ["vectors.npy", "labels.npy", "filenames.npy", "classnames.npy"]]
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
        data_detail = f"dossiers manquants: {', '.join(missing_splits)}" if missing_splits else "aucune image .jpg détectée dans data/out"

    missing_embedding_paths_count = 0
    embeddings_state = "Valide"
    embeddings_card_ok = not missing_embeddings
    embeddings_detail = (
        f"{len(required_embedding_files)} fichiers requis présents"
        if embeddings_card_ok else f"manquants: {', '.join(missing_embeddings)}"
    )

    filenames_path = emb_root / "filenames.npy"
    if embeddings_card_ok and data_ok and filenames_path.exists():
        stored_filenames = np.load(filenames_path, allow_pickle=True)
        for fp in stored_filenames:
            if not resolve_stored_path(fp).exists():
                missing_embedding_paths_count += 1
        # On contrôle ici non seulement la présence des artefacts `.npy`,
        # mais aussi leur validité "métier" : des embeddings peuvent exister
        # tout en pointant vers des images supprimées ou déplacées.
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
    # La section de statut remonte un diagnostic exploitable par l'équipe :
    # si l'upload est bloqué, l'utilisateur voit immédiatement si le problème
    # vient du modèle, des embeddings ou du dossier `data/out`.
    status = inspect_runtime_assets()
    rows = status["rows"]
    has_error = not bool(status["upload_enabled"])
    button_bg = "#fdecec" if has_error else "#eef6ff"
    button_fg = "#b42318" if has_error else "#175cd3"
    button_border = "#f04438" if has_error else "#84caff"

    st.markdown(
        f"""
        <style>
        div[data-testid="stExpander"] details {{ border: 1px solid {button_border}; border-radius: 12px; overflow: hidden; }}
        div[data-testid="stExpander"] details summary {{
            background: {button_bg} !important; background-color: {button_bg} !important;
            color: {button_fg} !important; -webkit-text-fill-color: {button_fg} !important; font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("État des ressources", expanded=False):
        cols = st.columns(len(rows))
        for col, row in zip(cols, rows):
            with col:
                st.markdown(
                    f"""
                    <div style="border-left: 6px solid {row['accent']}; background: {row['tone']}; border-radius: 12px; padding: 0.85rem 1rem; min-height: 128px;">
                        <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.35rem;">{row['label']}</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: {row['accent']}; margin-bottom: 0.35rem;">{row['state']}</div>
                        <div style="font-size: 0.92rem; line-height: 1.35;">{row['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if status["missing_embedding_paths_count"]:
            st.warning("Les embeddings sont désynchronisés avec `data/out`. L'upload est bloqué tant que les embeddings n'ont pas été régénérés.")
    return status
