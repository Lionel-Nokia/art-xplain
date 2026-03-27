from __future__ import annotations
# Active l'évaluation différée des annotations de type.
# Concrètement, cela permet d'écrire des annotations modernes comme list[str]
# sans que Python essaie de les résoudre immédiatement au moment du chargement.
# C'est utile pour :
# 1) améliorer la compatibilité selon les versions de Python,
# 2) éviter certains problèmes de références circulaires dans les types,
# 3) garder un code plus lisible avec les hints modernes.

import sys
import atexit
import asyncio
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from html import escape
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
from PIL import Image
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

from src.utils import ensure_dir, load_config, resolve_project_path, resolve_stored_path
# Importe une fonction utilitaire interne censée charger la configuration du projet.
# On l'utilise plus loin pour récupérer les chemins vers les embeddings et autres ressources.

from src.retrieval import StyleRetriever
# Importe la classe centrale du moteur de recherche de similarité stylistique.
# C'est elle qui va probablement encapsuler :
# - le chargement du modèle,
# - l'extraction d'embeddings,
# - la comparaison de similarité,
# - et éventuellement Grad-CAM.

try:
    from src.ia_agent import run_analysis
except Exception as exc:
    run_analysis = None
    IA_AGENT_IMPORT_ERROR = exc
else:
    IA_AGENT_IMPORT_ERROR = None


# =============================================================================
# CONFIGURATION GÉNÉRALE DE L'INTERFACE STREAMLIT
# =============================================================================

ASSETS_DIR = PROJECT_ROOT / "assets"
APP_LOGO_PATH = ASSETS_DIR / "artxplain-logo.svg"
INTERNAL_DF_DIR = PROJECT_ROOT / "data"
INTERNAL_DF_PATH = INTERNAL_DF_DIR / "internal_artworks.csv"
INTERNAL_DF_COLUMNS = ["artiste", "tableau", "style", "fichier", "analyse", "similarite"]
AI_GUIDE_PROFILE_NAME = "guide_musée"

# -----------------------------------------------------------------------------
# NOTE D'ARCHITECTURE
# -----------------------------------------------------------------------------
# Ce module mélange volontairement deux niveaux :
# 1. des helpers "métier / parsing / formatage" assez indépendants de Streamlit ;
# 2. une classe orchestratrice `ArtXplainApp` qui porte le flux UI de bout en bout.
#
# L'objectif du refactor n'est pas encore de découper le fichier en plusieurs modules,
# mais de rendre la lecture d'une revue de code plus simple :
# - les fonctions utilitaires restent globales car elles sont pures ou quasi-pures ;
# - la classe centralise l'état runtime et le pipeline d'affichage Streamlit ;
# - les dataclasses servent de "contrat de passage" entre étapes du pipeline.
#
# En pratique, la lecture recommandée du fichier est :
# - d'abord `ArtXplainApp.run()`
# - puis les dataclasses de contexte
# - puis les méthodes `_render_*` / `_build_*`
# - enfin les helpers globaux appelés par ces méthodes.
#
# MINI SOMMAIRE DE NAVIGATION
# - Helpers de formatage / parsing IA :
#   `_format_analysis_text`, `_extract_json_payload`, `_coerce_payload_to_chapters`,
#   `_match_artwork_analysis`, `_match_source_artwork_analysis`
# - Helpers retrieval / visualisation :
#   `get_retriever`, `_extract_artist_and_title`, `_build_style_names`,
#   `_select_explanation_layers`, `_build_random_gradcam_layer_numbers`
# - Chargement des artefacts :
#   `load_latent_and_meta`, `_inspect_runtime_assets`, `render_runtime_status`
# - Contrats de passage :
#   `AppConfig`, `QuerySource`, `ArtworkResultsContext`, `AIAnalysisState`
# - Orchestrateur UI :
#   `ArtXplainApp.run()` puis les méthodes privées regroupées par section
#   (cycle de vie, UI, IA, visualisation).


def _load_inline_svg(svg_path: Path) -> str:
    """
    Charge un SVG local et renvoie son contenu pour injection inline.
    """
    try:
        return svg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _compute_source_image_display_width(image_path: str) -> int:
    """
    Calcule une largeur d'affichage raisonnable selon les proportions de l'image source.
    """
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


def _empty_internal_dataframe() -> pd.DataFrame:
    """
    Construit le DataFrame interne avec son schéma cible.
    """
    return pd.DataFrame(columns=INTERNAL_DF_COLUMNS)


def _normalize_internal_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit la présence et l'ordre des colonnes attendues.
    """
    normalized = df.copy()
    for column in INTERNAL_DF_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    normalized = normalized[INTERNAL_DF_COLUMNS].fillna("")
    if "similarite" in normalized.columns:
        normalized["similarite"] = normalized["similarite"].apply(_normalize_similarity_json)
    return normalized.astype(str)


def _normalize_similarity_json(value: object) -> str:
    """
    Convertit la colonne similarite en JSON texte stable.
    """
    if value is None:
        return "[]"

    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)

    text = str(value).strip()
    if not text:
        return "[]"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return "[]"

    if not isinstance(parsed, list):
        return "[]"

    return json.dumps(parsed, ensure_ascii=False)


def _parse_similarity_json(value: object) -> list[dict[str, object]]:
    """
    Relit l'historique JSON de similarité pour une ligne.
    """
    normalized = _normalize_similarity_json(value)

    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    cleaned_history: list[dict[str, object]] = []
    for item in parsed:
        if isinstance(item, dict):
            cleaned_history.append(item)
    return cleaned_history


def _is_unknown_metadata(value: object) -> bool:
    """
    Détecte les métadonnées inutilisables pour le stockage.
    """
    return str(value).strip().lower() == "inconnu"


def _normalize_lookup_text(value: object) -> str:
    """
    Normalise un texte pour des comparaisons souples.
    """
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    for char in ["-", "–", "—", "_", "/", ",", ":", ";", ".", "(", ")", "'", '"']:
        text = text.replace(char, " ")
    return " ".join(text.split())


def _match_normalized_text(expected: str, candidate: str) -> bool:
    """
    Compare deux textes normalisés avec une logique souple.
    """
    if not expected or not candidate:
        return False

    if expected == candidate:
        return True

    if expected in candidate or candidate in expected:
        return True

    expected_tokens = expected.split()
    candidate_tokens = candidate.split()
    if expected_tokens and all(token in candidate_tokens for token in expected_tokens):
        return True

    return False


def _token_overlap_score(reference: str, candidate: str) -> float:
    """
    Calcule un score simple de recouvrement entre deux textes normalisés.
    """
    reference_tokens = set(reference.split())
    candidate_tokens = set(candidate.split())
    if not reference_tokens or not candidate_tokens:
        return 0.0

    overlap = reference_tokens.intersection(candidate_tokens)
    if not overlap:
        return 0.0

    return len(overlap) / max(len(reference_tokens), 1)


def _format_analysis_text(value: object) -> str:
    """
    Convertit une structure JSON en texte lisible.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return _strip_http_links(value.strip())
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_format_analysis_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = _format_analysis_text(item)
            if item_text:
                parts.append(f"{key} : {item_text}")
        return _strip_http_links("\n".join(parts))
    return _strip_http_links(str(value).strip())


def _analysis_text_to_html(text: str) -> str:
    """
    Transforme le texte d'analyse en HTML en mettant certains sous-titres en gras.
    """
    html = escape(str(text).strip())
    # REVIEW NOTE:
    # L'agent IA renvoie parfois du Markdown inline (`**...**`) à l'intérieur
    # d'un bloc que l'on affiche ensuite comme HTML. Sans conversion explicite,
    # les marqueurs apparaissent littéralement dans l'UI.
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    for subtitle in ["Contexte historique et technique", "Spécificités stylistiques"]:
        html = html.replace(subtitle, f"<strong>{subtitle}</strong>")
    return html.replace("\n", "<br>")


def _strip_http_links(text: str) -> str:
    """
    Supprime les liens HTTP/HTTPS du texte tout en conservant le contenu utile.
    """
    cleaned = str(text)
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"\((https?://[^)]+)\)", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\(\s+\)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_analysis_title(title: object) -> tuple[str, str]:
    """
    Extrait artiste et tableau depuis un titre de chapitre `Artiste - Tableau`.
    """
    title_text = str(title).strip()
    for separator in [" - ", " – ", " — "]:
        if separator in title_text:
            artist, artwork = title_text.split(separator, 1)
            return artist.strip(), artwork.strip()
    return "", title_text


def _is_global_analysis_title(title: object) -> bool:
    """
    Détecte un chapitre d'analyse globale / synthèse plutôt qu'une oeuvre précise.
    """
    normalized_title = _normalize_lookup_text(title)
    return any(
        marker in normalized_title
        for marker in [
            "proximite stylistique",
            "analyse stylistique commune",
            "comparaison stylistique globale",
            "analyse globale",
            "synthese stylistique",
        ]
    )


def _clean_analysis_subtitle(text: object) -> str:
    """
    Nettoie certains sous-titres génériques peu utiles à l'affichage.
    """
    subtitle = str(text).strip()
    normalized = _normalize_lookup_text(subtitle)
    if normalized in {
        "contexte historique et stylistique",
        "contexte historique",
    }:
        return ""
    return subtitle


def _format_chapter_content(chapter_content: object) -> str:
    """
    Formate le contenu d'un chapitre en concaténant `sous_titre` et `texte`.
    """
    if not isinstance(chapter_content, list):
        return _format_analysis_text(chapter_content)

    blocks: list[str] = []
    for item in chapter_content:
        if isinstance(item, dict):
            subtitle = _clean_analysis_subtitle(item.get("sous_titre", ""))
            text = str(item.get("texte", "")).strip()
            if subtitle and text:
                blocks.append(f"{subtitle}\n{text}")
            elif subtitle:
                blocks.append(subtitle)
            elif text:
                blocks.append(text)
        else:
            text = _format_analysis_text(item)
            if text:
                blocks.append(text)

    return "\n\n".join(block for block in blocks if block)


def _extract_json_payload(final_output: str) -> dict[str, object] | list[object] | None:
    """
    Extrait un JSON depuis une réponse brute, même si elle est entourée de fences Markdown.
    """
    raw_text = str(final_output).strip()
    candidates = [raw_text]

    if "```" in raw_text:
        fenced_blocks = raw_text.split("```")
        for block in fenced_blocks:
            cleaned = block.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            candidates.append(cleaned)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (dict, list)):
            return payload

    return None


def _coerce_payload_to_chapters(
    payload: dict[str, object] | list[object] | None,
) -> list[dict[str, object]] | None:
    """
    Convertit différents schémas JSON en une liste homogène de chapitres.
    """
    # REVIEW NOTE:
    # Cette fonction joue le rôle d'adaptateur de contrat entre les sorties IA
    # parfois hétérogènes et le format unique attendu par l'UI.
    # C'est volontairement l'un des rares endroits où l'on accepte plusieurs
    # schémas en entrée afin d'éviter de propager cette complexité dans le rendu.
    if payload is None:
        return None

    if isinstance(payload, dict):
        chapters = payload.get("chapitres")
        if isinstance(chapters, list):
            return [chapter for chapter in chapters if isinstance(chapter, dict)]

        works = payload.get("oeuvres")
        if isinstance(works, list):
            normalized_chapters: list[dict[str, object]] = []
            for work in works:
                if not isinstance(work, dict):
                    continue

                artist = str(work.get("artiste", "")).strip()
                title = str(work.get("titre", "")).strip()
                chapter_title = " - ".join(part for part in [artist, title] if part).strip()
                if not chapter_title:
                    chapter_title = title or artist or "Oeuvre"

                raw_analysis = work.get("analyse")
                if isinstance(raw_analysis, dict):
                    content = [
                        {
                            "sous_titre": str(key).replace("_", " ").strip().capitalize(),
                            "texte": _format_analysis_text(value),
                        }
                        for key, value in raw_analysis.items()
                        if _format_analysis_text(value)
                    ]
                else:
                    content = []
                    for key in [
                        "contexte_historique",
                        "specificites_stylistiques",
                        "elements_techniques_marquants",
                    ]:
                        value = _format_analysis_text(work.get(key))
                        if value:
                            content.append(
                                {
                                    "sous_titre": key.replace("_", " ").strip().capitalize(),
                                    "texte": value,
                                }
                            )

                if not content:
                    content = _format_analysis_text(work)

                normalized_chapters.append(
                    {
                        "titre": chapter_title,
                        "contenu": content,
                    }
                )

            global_candidates = [
                payload.get("comparaison_stylistique_globale"),
                payload.get("rapprochement_stylistique"),
                payload.get("analyse_stylistique_globale"),
            ]
            for global_candidate in global_candidates:
                global_text = _format_analysis_text(global_candidate)
                if global_text:
                    normalized_chapters.append(
                        {
                            "titre": "Analyse stylistique comparative",
                            "contenu": global_text,
                        }
                    )
                    break

            return normalized_chapters or None

    if isinstance(payload, list):
        return [chapter for chapter in payload if isinstance(chapter, dict)] or None

    return None


def _extract_chapters_payload(final_output: str) -> list[dict[str, object]] | None:
    """
    Relit le JSON et retourne la liste des chapitres si présente.
    """
    payload = _extract_json_payload(final_output)
    return _coerce_payload_to_chapters(payload)


def _match_artwork_analysis(
    final_output: str,
    artist: str,
    title: str,
    result_index: int | None = None,
) -> str | None:
    """
    Retrouve le meilleur fragment d'analyse pour une oeuvre donnée.
    """
    # REVIEW NOTE:
    # On ne s'appuie pas uniquement sur l'ordre des chapitres IA.
    # Le mapping est fait par score artiste / titre, car l'agent peut :
    # - changer l'ordre ;
    # - reformuler les titres ;
    # - inclure un chapitre global qui décale les positions.
    chapters = _extract_chapters_payload(final_output)
    title_key = _normalize_lookup_text(title)
    artist_key = _normalize_lookup_text(artist)
    if chapters is not None:
        best_score = 0.0
        best_content: str | None = None

        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            if _is_global_analysis_title(chapter.get("titre", "")):
                continue

            chapter_artist, chapter_title = _split_analysis_title(chapter.get("titre", ""))
            normalized_chapter_artist = _normalize_lookup_text(chapter_artist)
            normalized_chapter_title = _normalize_lookup_text(chapter_title)

            if (
                _match_normalized_text(artist_key, normalized_chapter_artist)
                and _match_normalized_text(title_key, normalized_chapter_title)
            ):
                return _format_chapter_content(chapter.get("contenu", []))

            score = 0.0
            if title_key:
                score += max(
                    _token_overlap_score(title_key, normalized_chapter_title),
                    _token_overlap_score(title_key, _normalize_lookup_text(chapter.get("titre", ""))),
                )
            if artist_key:
                artist_score = max(
                    _token_overlap_score(artist_key, normalized_chapter_artist),
                    _token_overlap_score(artist_key, _normalize_lookup_text(chapter.get("titre", ""))),
                )
                score += min(artist_score, 1.0)

            if chapter_artist and chapter_title:
                score += 0.1

            if score > best_score:
                best_score = score
                best_content = _format_chapter_content(chapter.get("contenu", []))

        if best_score >= 0.8:
            return best_content

    return None


def _extract_global_analysis(final_output: str) -> str | None:
    """
    Retrouve la section de comparaison stylistique globale.
    """
    chapters = _extract_chapters_payload(final_output)
    if chapters is not None:
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            if _is_global_analysis_title(chapter.get("titre", "")):
                return _format_chapter_content(chapter.get("contenu", []))

    return None


def _match_source_artwork_analysis(
    final_output: str,
    source_artist: str,
    source_title: str,
    artwork_of_interest: str,
    source_display_name: str | None = None,
) -> str | None:
    """
    Retrouve l'analyse de l'oeuvre d'intérêt sans jamais tomber sur un tableau suggéré.
    """
    direct_match = _match_artwork_analysis(final_output, source_artist, source_title)
    if direct_match:
        return direct_match

    chapters = _extract_chapters_payload(final_output)
    if chapters is None:
        return None

    source_title_key = _normalize_lookup_text(source_title)
    interest_key = _normalize_lookup_text(artwork_of_interest)
    display_key = _normalize_lookup_text(Path(str(source_display_name or "")).stem)
    source_artist_key = _normalize_lookup_text(source_artist)

    candidate_keys = [
        key for key in [source_title_key, interest_key, display_key] if key
    ]

    best_score = 0.0
    best_content: str | None = None

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue

        chapter_title = str(chapter.get("titre", "")).strip()
        normalized_chapter_title = _normalize_lookup_text(chapter_title)

        if _is_global_analysis_title(chapter_title):
            continue

        chapter_artist, chapter_artwork = _split_analysis_title(chapter_title)
        normalized_chapter_artist = _normalize_lookup_text(chapter_artist)
        normalized_chapter_artwork = _normalize_lookup_text(chapter_artwork)

        if any(
            _match_normalized_text(candidate_key, normalized_chapter_title)
            or _match_normalized_text(candidate_key, normalized_chapter_artwork)
            for candidate_key in candidate_keys
        ):
            return _format_chapter_content(chapter.get("contenu", []))

        score = 0.0
        for candidate_key in candidate_keys:
            score = max(
                score,
                _token_overlap_score(candidate_key, normalized_chapter_title),
                _token_overlap_score(candidate_key, normalized_chapter_artwork),
            )

        if source_artist_key and normalized_chapter_artist:
            if _match_normalized_text(source_artist_key, normalized_chapter_artist):
                score += 0.2
            else:
                score += min(
                    _token_overlap_score(source_artist_key, normalized_chapter_artist),
                    0.2,
                )

        if score > best_score:
            best_score = score
            best_content = _format_chapter_content(chapter.get("contenu", []))

    if best_score >= 0.55:
        return best_content

    return None


def _run_async_analysis_sync(
    candidates_df: pd.DataFrame,
    artwork_of_interest: str,
    profile_name: str,
) -> dict[str, str]:
    """
    Exécute l'analyse async dans un contexte synchrone Streamlit.
    """
    if run_analysis is None:
        if IA_AGENT_IMPORT_ERROR is not None:
            raise RuntimeError(f"Import ia_agent impossible : {IA_AGENT_IMPORT_ERROR}")
        raise RuntimeError("Le module ia_agent n'est pas disponible.")

    def _build_coroutine():
        return run_analysis(
            df=candidates_df,
            artwork_of_interest=artwork_of_interest,
            config_path=str(PROJECT_ROOT / "config" / "config_agent.yaml"),
            profile_name=profile_name,
            output_folder=str(PROJECT_ROOT / "outputs"),
            save_to_file=False,
        )

    try:
        analysis = asyncio.run(_build_coroutine())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            analysis = loop.run_until_complete(_build_coroutine())
        finally:
            loop.close()

    final_output = str(analysis["result"].final_output).strip()
    return {
        "final_output": final_output,
        "output_path": str(analysis.get("output_path", "")),
    }


@st.cache_data(show_spinner=False)
def _get_cached_ai_analysis(
    candidates_json: str,
    artwork_of_interest: str,
    profile_name: str,
) -> dict[str, str]:
    """
    Met en cache l'analyse OpenAI pour éviter les relances inutiles.
    """
    candidates_df = pd.DataFrame(json.loads(candidates_json))
    return _run_async_analysis_sync(candidates_df, artwork_of_interest, profile_name)


@st.cache_data(show_spinner=False)
def _is_ai_agent_enabled() -> bool:
    """
    Lit le flag d'activation de l'agent IA depuis la configuration.
    """
    try:
        config = load_config("config/config_agent.yaml")
    except Exception:
        return False

    ai_config = config.get("ai-agent", {})
    return bool(ai_config.get("ai_active", True))


@st.cache_data(show_spinner=False)
def _get_default_ai_profile_name() -> str:
    """
    Lit le nom du profil IA par défaut pour l'afficher dans l'interface.
    """
    try:
        config = load_config("config/config_agent.yaml")
    except Exception:
        return AI_GUIDE_PROFILE_NAME

    ai_config = config.get("ai-agent", {})
    profile_name = str(ai_config.get("default_profile", "")).strip()
    return profile_name or AI_GUIDE_PROFILE_NAME


@st.cache_data(show_spinner=False)
def _get_available_ai_profile_names() -> list[str]:
    """
    Retourne la liste des profils IA disponibles dans la configuration.
    """
    try:
        config = load_config("config/config_agent.yaml")
    except Exception:
        return [AI_GUIDE_PROFILE_NAME]

    ai_config = config.get("ai-agent", {})
    profiles = ai_config.get("profiles", {})
    if isinstance(profiles, dict):
        names = [str(name).strip() for name in profiles.keys() if str(name).strip()]
        if names:
            return names

    return [AI_GUIDE_PROFILE_NAME]


def _load_internal_dataframe() -> pd.DataFrame:
    """
    Relit le DataFrame interne depuis le disque au démarrage.
    """
    if not INTERNAL_DF_PATH.exists():
        return _empty_internal_dataframe()

    try:
        loaded = pd.read_csv(INTERNAL_DF_PATH)
        return _normalize_internal_dataframe(loaded)
    except Exception:
        return _empty_internal_dataframe()


def _save_internal_dataframe(df: pd.DataFrame) -> None:
    """
    Sauvegarde le DataFrame interne sur le disque.
    """
    ensure_dir(INTERNAL_DF_DIR)
    _normalize_internal_dataframe(df).to_csv(INTERNAL_DF_PATH, index=False)


def _register_internal_dataframe_shutdown_hook() -> None:
    """
    Sauvegarde de secours à l'arrêt du processus Streamlit.
    """
    if st.session_state.get("_internal_dataframe_shutdown_hook_registered"):
        return

    def _persist_on_exit() -> None:
        df = st.session_state.get("internal_artworks_df")
        if isinstance(df, pd.DataFrame):
            try:
                _save_internal_dataframe(df)
            except Exception:
                pass

    atexit.register(_persist_on_exit)
    st.session_state["_internal_dataframe_shutdown_hook_registered"] = True


def _initialize_internal_dataframe() -> None:
    """
    Charge le DataFrame interne une seule fois par session Streamlit.
    """
    if "internal_artworks_df" not in st.session_state:
        st.session_state["internal_artworks_df"] = _load_internal_dataframe()

    _register_internal_dataframe_shutdown_hook()


def _update_internal_dataframe_from_results(
    df_results: pd.DataFrame,
    source_artist: str,
    source_title: str,
) -> pd.DataFrame:
    """
    Ajoute les artistes absents du DataFrame interne et historise les similarités.
    """
    internal_df = _normalize_internal_dataframe(st.session_state["internal_artworks_df"])

    if _is_unknown_metadata(source_artist) or _is_unknown_metadata(source_title):
        st.session_state["internal_artworks_df"] = internal_df
        return internal_df

    source_df = (
        df_results.loc[:, ["artiste", "tableau", "style", "fichier", "similarité"]]
        .copy()
        .fillna("")
        .astype(str)
    )
    source_df = source_df[
        ~source_df["artiste"].apply(_is_unknown_metadata)
        & ~source_df["tableau"].apply(_is_unknown_metadata)
    ].copy()
    source_df["analyse"] = ""
    source_df["similarite"] = "[]"
    source_df = source_df.rename(columns={"similarité": "similarite_courante"})

    if source_df.empty:
        st.session_state["internal_artworks_df"] = internal_df
        return internal_df

    existing_pairs = set(
        zip(
            internal_df["artiste"].astype(str).str.strip(),
            internal_df["tableau"].astype(str).str.strip(),
        )
    )
    existing_keys = set(
        zip(
            internal_df["artiste"].astype(str).str.strip(),
            internal_df["tableau"].astype(str).str.strip(),
            internal_df["fichier"].astype(str).str.strip(),
        )
    )
    source_df["pair_key"] = list(
        zip(
            source_df["artiste"].astype(str).str.strip(),
            source_df["tableau"].astype(str).str.strip(),
        )
    )

    rows_to_add = source_df[~source_df["pair_key"].isin(existing_pairs)].copy()

    if not rows_to_add.empty:
        rows_to_add["similarite"] = rows_to_add["similarite_courante"].apply(
            lambda similarity: json.dumps(
                [
                    {
                        "artiste_source": source_artist,
                        "tableau_source": source_title,
                        "similarite": float(similarity),
                    }
                ],
                ensure_ascii=False,
            )
        )
        rows_to_add = rows_to_add[INTERNAL_DF_COLUMNS]
        internal_df = pd.concat([internal_df, rows_to_add], ignore_index=True)

    results_by_key = {
        (
            str(row["artiste"]).strip(),
            str(row["tableau"]).strip(),
            str(row["fichier"]).strip(),
        ): float(row["similarite_courante"])
        for _, row in source_df.iterrows()
    }

    source_signature = (
        str(source_artist).strip(),
        str(source_title).strip(),
    )

    for idx, row in internal_df.iterrows():
        key = (
            str(row["artiste"]).strip(),
            str(row["tableau"]).strip(),
            str(row["fichier"]).strip(),
        )
        if key not in existing_keys:
            continue
        similarity_value = results_by_key.get(key)
        if similarity_value is None:
            continue

        history = _parse_similarity_json(row["similarite"])
        history_signatures = {
            (
                str(item.get("artiste_source", "")).strip(),
                str(item.get("tableau_source", "")).strip(),
            )
            for item in history
        }
        if source_signature in history_signatures:
            continue

        history.append(
            {
                "artiste_source": source_artist,
                "tableau_source": source_title,
                "similarite": similarity_value,
            }
        )
        internal_df.at[idx, "similarite"] = json.dumps(history, ensure_ascii=False)

    internal_df = _normalize_internal_dataframe(internal_df)
    st.session_state["internal_artworks_df"] = internal_df
    _save_internal_dataframe(internal_df)

    return st.session_state["internal_artworks_df"]

st.set_page_config(page_title="Art-Xplain", layout="wide")
# Configure la page Streamlit avant tout rendu visuel important.
# page_title : titre de l'onglet du navigateur.
# layout="wide" : utilise toute la largeur disponible, ce qui est très utile
# pour afficher plusieurs images côte à côte et une visualisation UMAP large.

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
        --museum-shadow: 0 14px 34px rgba(17, 17, 17, 0.06);
        --museum-font: "Georgia", "Iowan Old Style", "Times New Roman", serif;
    }

    html, body, [class*="css"] {
        font-family: var(--museum-font);
        color-scheme: light;
    }

    .stApp {
        color: var(--museum-ink);
        background: #ffffff;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(17, 17, 17, 0.08);
    }

    [data-testid="block-container"] {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        font-family: var(--museum-font);
        color: var(--museum-ink);
        letter-spacing: 0.01em;
    }

    h3 {
        font-size: 1.6rem;
        margin-top: 1.4rem;
    }

    p, li, label, .stCaption {
        color: var(--museum-muted);
    }

    .museum-hero {
        padding: 2.2rem 2rem 1.9rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--museum-line);
        border-radius: 10px;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(246, 243, 238, 0.78)),
            linear-gradient(90deg, rgba(255, 255, 255, 0.38), rgba(255, 255, 255, 0.12)),
            url("https://www.mam.paris.fr/sites/default/files/home/bandeau/hall_0.png");
        background-size: cover;
        background-position: center center;
        box-shadow: var(--museum-shadow);
        position: relative;
        overflow: hidden;
    }

    .museum-hero-header {
        display: flex;
        align-items: flex-start;
        gap: 1.3rem;
    }

    .museum-logo-wrap {
        width: 176px;
        min-width: 176px;
        padding-top: 0.2rem;
    }

    .museum-logo-wrap svg {
        display: block;
        width: 100%;
        height: auto;
    }

    .museum-hero-copy {
        flex: 1;
        min-width: 0;
    }

    .museum-hero::after {
        content: "";
        position: absolute;
        left: 2rem;
        right: 2rem;
        top: 0.95rem;
        height: 4px;
        background: var(--museum-accent);
    }

    .museum-kicker {
        display: inline-block;
        margin-bottom: 0.9rem;
        padding: 0;
        border-radius: 0;
        background: transparent;
        color: var(--museum-accent);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .museum-title {
        margin: 0.15rem 0 0 0;
        font-size: clamp(2.6rem, 4vw, 4.9rem);
        line-height: 0.96;
    }

    .museum-lead {
        max-width: 52rem;
        margin: 1rem 0 0 0;
        font-size: 1rem;
        line-height: 1.72;
        color: var(--museum-muted);
    }

    .museum-meta {
        margin-top: 1.4rem;
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--museum-muted);
    }

    @media (max-width: 820px) {
        .museum-hero-header {
            flex-direction: column;
            gap: 0.9rem;
        }

        .museum-logo-wrap {
            width: 152px;
            min-width: 152px;
        }
    }

    .museum-card {
        padding: 1rem 1.1rem;
        margin: 0.45rem 0 0.8rem 0;
        border: 1px solid var(--museum-line);
        border-radius: 8px;
        background: var(--museum-panel);
        box-shadow: var(--museum-shadow);
    }

    .museum-card strong {
        display: block;
        color: var(--museum-ink);
        font-family: var(--museum-font);
        font-size: 1.08rem;
        margin-bottom: 0.15rem;
    }

    .museum-card em {
        color: var(--museum-muted);
        font-size: 0.96rem;
    }

    .museum-style-chip {
        display: inline-block;
        margin: 0.15rem 0 1rem 0;
        padding: 0.48rem 0.88rem;
        border-radius: 4px;
        border: 1px solid rgba(177, 34, 28, 0.22);
        background: var(--museum-accent-soft);
        color: var(--museum-accent);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stExpander"],
    [data-testid="stDataFrame"],
    [data-testid="stAlert"] {
        border: 1px solid var(--museum-line) !important;
        border-radius: 8px !important;
        background: var(--museum-panel) !important;
        box-shadow: var(--museum-shadow);
    }

    [data-testid="stFileUploaderDropzone"] {
        background:
            linear-gradient(180deg, rgba(177, 34, 28, 0.03), rgba(255, 255, 255, 0.98)) !important;
        border-style: dashed !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] section button {
        -webkit-appearance: none !important;
        appearance: none !important;
        -webkit-tap-highlight-color: transparent;
        border-radius: 999px !important;
        border: 1px solid rgba(177, 34, 28, 0.22) !important;
        background-color: rgba(255, 255, 255, 0.92) !important;
        background-image: none !important;
        background: rgba(255, 255, 255, 0.92) !important;
        color: var(--museum-accent) !important;
        -webkit-text-fill-color: var(--museum-accent) !important;
        box-shadow: none !important;
        filter: none !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:focus,
    [data-testid="stFileUploader"] button:focus-visible,
    [data-testid="stFileUploader"] button:active,
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:focus,
    [data-testid="stFileUploaderDropzone"] button:focus-visible,
    [data-testid="stFileUploaderDropzone"] button:active,
    [data-testid="stFileUploader"] section button:hover,
    [data-testid="stFileUploader"] section button:focus,
    [data-testid="stFileUploader"] section button:focus-visible,
    [data-testid="stFileUploader"] section button:active {
        border-color: rgba(177, 34, 28, 0.34) !important;
        background-color: rgba(177, 34, 28, 0.08) !important;
        background-image: none !important;
        background: rgba(177, 34, 28, 0.08) !important;
        color: var(--museum-accent) !important;
        -webkit-text-fill-color: var(--museum-accent) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stButton"] > button,
    button[data-testid^="stBaseButton"],
    button[kind="secondary"],
    [data-baseweb="button"] {
        -webkit-appearance: none;
        appearance: none;
        -webkit-tap-highlight-color: transparent;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        text-align: left;
        border-radius: 4px;
        border: 1px solid rgba(17, 17, 17, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.72) !important;
        background-image: none !important;
        background: rgba(255, 255, 255, 0.72) !important;
        color: var(--museum-ink) !important;
        -webkit-text-fill-color: var(--museum-ink) !important;
        padding: 0.28rem 0.5rem;
        min-height: 1.7rem;
        box-shadow: none !important;
        opacity: 1 !important;
        filter: none !important;
        -webkit-box-shadow: none !important;
        -webkit-border-image: none !important;
        transition: background 180ms ease, border-color 180ms ease, color 180ms ease;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:focus-visible,
    div[data-testid="stButton"] > button:active,
    button[data-testid^="stBaseButton"]:hover,
    button[data-testid^="stBaseButton"]:focus,
    button[data-testid^="stBaseButton"]:focus-visible,
    button[data-testid^="stBaseButton"]:active,
    button[kind="secondary"]:hover,
    button[kind="secondary"]:focus,
    button[kind="secondary"]:focus-visible,
    button[kind="secondary"]:active,
    [data-baseweb="button"]:hover,
    [data-baseweb="button"]:focus,
    [data-baseweb="button"]:focus-visible,
    [data-baseweb="button"]:active {
        border-color: rgba(177, 34, 28, 0.28);
        background-color: rgba(177, 34, 28, 0.05) !important;
        background-image: none !important;
        background: rgba(177, 34, 28, 0.05) !important;
        color: var(--museum-accent) !important;
        -webkit-text-fill-color: var(--museum-accent) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stButton"] > button *,
    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span,
    div[data-testid="stButton"] > button em,
    div[data-testid="stButton"] > button strong,
    button[data-testid^="stBaseButton"] *,
    button[kind="secondary"] *,
    [data-baseweb="button"] * {
        color: inherit !important;
        -webkit-text-fill-color: inherit !important;
        background: transparent !important;
    }

    div[data-testid="stButton"] > button p {
        color: inherit;
        width: 100%;
        margin: 0;
        text-align: left;
        font-size: 0.88rem;
        line-height: 1.1;
    }

    [data-testid="stCheckbox"] label,
    [data-testid="stFileUploader"] label {
        color: var(--museum-ink);
        font-weight: 600;
    }

    [data-testid="stImage"] img {
        border-radius: 4px;
        border: 1px solid var(--museum-line);
        box-shadow: 0 16px 32px rgba(16, 16, 16, 0.10);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--museum-muted);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: var(--museum-panel);
        border: 1px solid var(--museum-line);
    }

    .stMarkdown a {
        color: var(--museum-accent);
    }

    [data-testid="stImage"] img,
    [data-testid="stImage"] picture,
    [data-testid="stImage"] > div {
        max-width: 100%;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        overflow-x: auto;
    }

    div[data-testid="stButton"] > button p {
        overflow-wrap: anywhere;
    }

    @media (max-width: 900px) {
        [data-testid="block-container"] {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .museum-hero {
            padding: 1.35rem 1rem 1.15rem 1rem;
            border-radius: 8px;
            background-position: 62% center;
        }

        .museum-hero::after {
            left: 1rem;
            right: 1rem;
            top: 0.7rem;
        }

        .museum-title {
            font-size: clamp(2rem, 10vw, 3.1rem);
            line-height: 1;
        }

        .museum-kicker,
        .museum-meta,
        .museum-style-chip {
            letter-spacing: 0.08em;
        }

        .museum-lead {
            font-size: 0.96rem;
            line-height: 1.55;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0.85rem;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }

        [data-testid="stImage"],
        [data-testid="stImage"] > div,
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }

        .js-plotly-plot,
        .plot-container,
        .svg-container {
            width: 100% !important;
            max-width: 100% !important;
        }

        [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
            max-width: 100% !important;
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stButton"] > button,
        button[data-testid^="stBaseButton"],
        button[kind="secondary"],
        [data-baseweb="button"] {
            min-height: 2.4rem;
            padding: 0.45rem 0.7rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.92rem;
        }

        .museum-card {
            padding: 0.85rem 0.9rem;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"] {
            overflow-wrap: anywhere;
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
            <div class="museum-logo-wrap">{_load_inline_svg(APP_LOGO_PATH)}</div>
            <div class="museum-hero-copy">
                <span class="museum-kicker">Musee d'Art Numérique & Datascience</span>
                <h1 class="museum-title">Art-Xplain</h1>
                <p class="museum-lead">
                    Exploration des proximités
                    stylistiques entre œuvres, à partir d'une image source, de voisinages visuels
                    et d'une lecture analytique par embeddings, UMAP et Grad-CAM++.
                </p>
                <div class="museum-meta">Collections numeriques • Similarite stylistique • Analyse visuelle</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

_initialize_internal_dataframe()


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


def _build_random_gradcam_layer_numbers(
    pair_count: int,
    min_layer: int = 1,
    max_layer: int = 245,
) -> list[int]:
    """
    Génère des couches réparties sur l'intervalle complet avec une part d'aléatoire.
    """
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


# =============================================================================
# CHARGEMENT DES DONNÉES LATENTES (UMAP + MÉTADONNÉES)
# =============================================================================

@st.cache_data
def _load_umap_bundle(bundle_path: str) -> dict[str, np.ndarray]:
    """
    Charge le bundle UMAP compressé et retourne ses tableaux.
    """
    with np.load(bundle_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


@st.cache_data
def _load_numpy_array(path: str, allow_pickle: bool = False) -> np.ndarray:
    """
    Charge un tableau NumPy depuis le disque avec mise en cache Streamlit.
    """
    return np.load(path, allow_pickle=allow_pickle)


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

    progress_text = st.empty()
    progress_bar = st.progress(0)

    def update_progress(step: int, total: int, message: str) -> None:
        progress_text.caption(f"Chargement des espaces latents : {message}")
        progress_bar.progress(step / total)

    bundle_path = emb_root / "umap_bundle.npz"
    # Construit le chemin vers un fichier compressé .npz contenant potentiellement
    # toutes les données nécessaires à l'affichage UMAP dans un seul bundle.

    try:
        if bundle_path.exists():
            update_progress(1, 5, "ouverture du bundle UMAP")
            data = _load_umap_bundle(str(bundle_path))
        # Si le bundle existe, on le charge.
        # np.load(..., allow_pickle=True) permet de lire des objets Python sérialisés,
        # ce qui peut être nécessaire pour classnames ou filenames.

            required_keys = {"latent_2d", "labels", "classnames", "filenames"}
        # Ensemble des clés indispensables pour que l'UMAP fonctionne correctement.

            update_progress(2, 5, "vérification du contenu du bundle")
            missing = required_keys.difference(set(data.keys()))
        # data.files contient les clés présentes dans le fichier .npz.
        # On calcule celles qui manquent.

            if missing:
                raise ValueError(
                    f"Le bundle UMAP est incomplet. Clés manquantes: {sorted(missing)}"
                )
            # On arrête explicitement avec un message clair plutôt que de laisser
            # l'application échouer plus loin de manière obscure.

            update_progress(3, 5, "lecture de la projection UMAP")
            latent_2d = np.asarray(data["latent_2d"])
        # Projection 2D des embeddings, typiquement shape (N, 2).

            update_progress(4, 5, "lecture des labels et métadonnées")
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
                progress_text.empty()
                progress_bar.empty()
                return None
            # Si même la projection principale n'existe pas, on considère que l'UMAP
            # n'est pas disponible et on renvoie None.

            update_progress(1, 5, "lecture de la projection UMAP")
            latent_2d = _load_numpy_array(str(latent_path))
        # Charge la projection 2D.

            update_progress(2, 5, "lecture des labels")
            labels = _load_numpy_array(str(emb_root / "labels.npy"))
        # Charge les labels.

            update_progress(3, 5, "lecture des noms de styles")
            classnames = _load_numpy_array(str(emb_root / "classnames.npy"), allow_pickle=True)
        # Charge les noms de classes.

            update_progress(4, 5, "lecture des chemins d'oeuvres")
            filenames = _load_numpy_array(str(emb_root / "filenames.npy"), allow_pickle=True)
        # Charge les chemins de fichiers.

            latent_2d = np.asarray(latent_2d)
            labels = np.asarray(labels)
            classnames = _coerce_object_array(classnames)
            filenames = _coerce_object_array([str(resolve_stored_path(fp)) for fp in filenames])
        # Uniformisation des types pour la suite du traitement.

        update_progress(5, 5, "validation finale")

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
    finally:
        progress_text.empty()
        progress_bar.empty()


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
            -webkit-appearance: none;
            appearance: none;
            background: {button_bg} !important;
            background-color: {button_bg} !important;
            color: {button_fg} !important;
            -webkit-text-fill-color: {button_fg} !important;
            font-weight: 700;
        }}
        div[data-testid="stExpander"] details summary:hover,
        div[data-testid="stExpander"] details summary:focus,
        div[data-testid="stExpander"] details summary:focus-visible,
        div[data-testid="stExpander"] details summary:active {{
            background: {button_bg} !important;
            background-color: {button_bg} !important;
            color: {button_fg} !important;
            -webkit-text-fill-color: {button_fg} !important;
            outline: none !important;
            box-shadow: none !important;
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


@dataclass
class AppConfig:
    """
    Regroupe les options choisies par l'utilisateur dans le panneau Configuration.

    Ce conteneur évite de faire circuler plusieurs variables primitives (`k`,
    profil IA, nombre de paires Grad-CAM, etc.) à travers tout le pipeline.
    En revue de code, il sert de "snapshot" de la configuration effective.
    """
    result_count: int
    ai_profile_name: str
    gradcam_pair_count: int
    ai_agent_enabled: bool


@dataclass
class QuerySource:
    """
    Représente l'oeuvre source réellement utilisée pour une exécution donnée.

    La source peut venir soit :
    - d'un upload utilisateur ;
    - d'un rechargement depuis la galerie de résultats.
    """
    query_path: str | None
    source_display_name: str | None
    source_notice: str | None


@dataclass
class ArtworkResultsContext:
    """
    Contexte principal construit après le retrieval.

    Cette structure centralise tout ce qui est dérivé de la recherche :
    - oeuvre source interprétée ;
    - résultats bruts du retriever ;
    - version tabulaire pour affichage ;
    - raccourcis vers top-1 / top-2 / top-3.
    """
    query_path: str
    source_display_name: str | None
    source_artist: str
    source_title: str
    results: list[dict[str, object]]
    rows: list[dict[str, object]]
    df_results: pd.DataFrame
    best: dict[str, object]
    second_best: dict[str, object] | None
    third_best: dict[str, object] | None


@dataclass
class AIAnalysisState:
    """
    État de la phase d'analyse IA pour le rendu courant.

    On sépare ce bloc du `ArtworkResultsContext` car l'analyse IA est optionnelle,
    coûteuse et potentiellement en erreur. Cette séparation aide à raisonner
    explicitement sur les cas :
    - IA désactivée ;
    - IA activée mais indisponible ;
    - IA activée avec payload exploitable.
    """
    enabled: bool
    profile_name: str
    payload: dict[str, str] | None
    error: str | None
    artwork_of_interest: str
    candidates_df: pd.DataFrame


class ArtXplainApp:
    """
    Orchestrateur principal de l'application Streamlit.

    La classe ne cherche pas à encapsuler toute la logique métier du fichier.
    Son rôle est plus ciblé :
    - charger les ressources techniques (retriever, bundle UMAP) ;
    - gérer l'état UI / session Streamlit ;
    - dérouler un pipeline de rendu lisible et stable.
    """
    def __init__(self) -> None:
        self.retriever: StyleRetriever | None = None
        self.retriever_error: Exception | None = None
        self.latent_bundle = None
        self.latent_error: Exception | None = None
        self.runtime_status: dict[str, object] | None = None
        self.available_ai_profiles: list[str] = []
        self.default_ai_profile_name = AI_GUIDE_PROFILE_NAME

    # ---------------------------------------------------------------------
    # SECTION 1 - CYCLE DE VIE GLOBAL DE L'APPLICATION
    # ---------------------------------------------------------------------
    def run(self) -> None:
        """
        Point d'entrée applicatif.

        Le flux est volontairement linéaire pour faciliter la revue :
        1. chargement des ressources lourdes ;
        2. lecture / initialisation de l'état Streamlit ;
        3. résolution de la source courante ;
        4. retrieval ;
        5. rendu des sections UI dépendantes du contexte obtenu.
        """
        self._load_resources()
        self.runtime_status = render_runtime_status()
        self.available_ai_profiles = _get_available_ai_profile_names()
        self.default_ai_profile_name = _get_default_ai_profile_name()

        if self.retriever_error is not None:
            st.warning(f"Moteur de retrieval indisponible : {self.retriever_error}")

        self._initialize_session_state()
        uploaded = self._render_uploader()
        config = self._render_configuration_panel()

        if self.retriever is None:
            self._render_unavailable_state(uploaded)
            return

        source = self._resolve_query_source(uploaded)
        if source.query_path is None:
            st.stop()

        if source.source_notice:
            st.info(source.source_notice)

        results_context = self._build_results_context(source, config.result_count)
        self._render_source_section(results_context)
        ai_state = self._build_ai_analysis_state(results_context, config)
        self._render_source_ai_section(results_context, ai_state)
        self._render_visual_comparison(results_context)
        self._render_global_ai_section(results_context, ai_state)
        self._render_summary(results_context.df_results)
        self._render_umap(results_context)
        self._render_gradcam(results_context, config.gradcam_pair_count)

    def _load_resources(self) -> None:
        """
        Charge les dépendances coûteuses mais non bloquantes séparément.

        On capture les erreurs au lieu de laisser planter le script Streamlit,
        ce qui permet de conserver une UI partiellement fonctionnelle avec des
        messages d'état explicites.
        """
        try:
            self.retriever = get_retriever()
        except Exception as exc:
            self.retriever_error = exc

        try:
            self.latent_bundle = load_latent_and_meta()
        except Exception as exc:
            self.latent_error = exc

    def _initialize_session_state(self) -> None:
        """
        Normalise les clés de session utilisées comme toggles de rendu.

        Streamlit relance le script complet à chaque interaction ; cette méthode
        joue donc un rôle essentiel pour transformer les `st.session_state[...]`
        en source de vérité stable entre deux reruns.
        """
        if "show_gradcam_history" not in st.session_state:
            st.session_state["show_gradcam_history"] = False

        if st.session_state.pop("reset_gradcam_history", False):
            st.session_state["show_gradcam_history"] = False

        if st.session_state.pop("reset_ai_analyses", False):
            st.session_state["show_ai_analyses"] = False

    # ---------------------------------------------------------------------
    # SECTION 2 - CONTRÔLES D'ENTRÉE ET CONFIGURATION UI
    # ---------------------------------------------------------------------
    def _render_uploader(self):
        """
        Affiche le widget d'upload et détecte un éventuel changement de fichier.

        Le changement d'upload réinitialise certains panneaux coûteux ou
        contextuels (Grad-CAM, analyses IA), afin d'éviter d'afficher des
        résultats devenus incohérents avec la nouvelle oeuvre source.
        """
        upload_disabled = bool(
            (self.retriever_error is not None)
            or (not bool(self.runtime_status and self.runtime_status["upload_enabled"]))
        )

        uploaded = st.file_uploader(
            "Upload une image (jpg/png/webp)",
            type=["jpg", "jpeg", "png", "webp"],
            disabled=upload_disabled,
        )

        uploaded_signature = None
        if uploaded is not None:
            uploaded_signature = f"{uploaded.name}:{uploaded.size}"

        if uploaded_signature != st.session_state.get("uploaded_signature"):
            st.session_state["uploaded_signature"] = uploaded_signature
            st.session_state["show_gradcam_history"] = False
            st.session_state["show_ai_analyses"] = False
            if uploaded_signature is not None:
                st.session_state["source_mode"] = "uploaded"

        return uploaded

    def _render_configuration_panel(self) -> AppConfig:
        """
        Rend le panneau de configuration utilisateur et retourne un objet typé.

        Le retour sous forme de dataclass rend le reste du code plus lisible
        qu'une succession de variables globales dispersées.
        """
        ai_agent_enabled = _is_ai_agent_enabled()

        with st.expander("Configuration", expanded=False):
            selected_result_count = st.slider(
                "Nombre de tableaux comparés",
                min_value=1,
                max_value=4,
                value=4,
                key="result_count",
                help="Définit combien d'oeuvres similaires afficher et comparer.",
            )

            selected_ai_profile = st.selectbox(
                "Profil de l'agent IA",
                options=self.available_ai_profiles,
                index=(
                    self.available_ai_profiles.index(self.default_ai_profile_name)
                    if self.default_ai_profile_name in self.available_ai_profiles
                    else 0
                ),
                key="selected_ai_profile",
                disabled=not ai_agent_enabled,
                help="Choisit le style de commentaire utilisé pour les analyses IA.",
            )

            gradcam_pair_count = st.slider(
                "Nombre de paires Grad-CAM",
                min_value=1,
                max_value=30,
                value=10,
                key="gradcam_pair_count",
                help="Définit combien de paires de cartes Grad-CAM afficher.",
            )

            if not ai_agent_enabled:
                st.caption("L'agent IA est désactivé dans la configuration actuelle.")

        profile_name = str(selected_ai_profile).strip() or self.default_ai_profile_name
        return AppConfig(
            result_count=int(selected_result_count),
            ai_profile_name=profile_name,
            gradcam_pair_count=int(gradcam_pair_count),
            ai_agent_enabled=ai_agent_enabled,
        )

    def _resolve_query_source(self, uploaded) -> QuerySource:
        """
        Détermine quelle image sert de source pour cette exécution.

        Priorité actuelle :
        - une oeuvre recliquée depuis la galerie ;
        - sinon l'upload courant.
        """
        # REVIEW NOTE:
        # Le `session_state["source_mode"] == "gallery"` donne priorité à une
        # oeuvre rechargée depuis les résultats, même si un upload existe encore
        # dans le widget Streamlit. Cela évite que le rerun revienne par erreur
        # à l'ancien fichier uploadé après un clic dans la galerie.
        query_path = None
        source_display_name = None
        source_notice = None

        if (
            st.session_state.get("source_mode") == "gallery"
            and st.session_state.get("source_image_path")
        ):
            query_path = str(st.session_state["source_image_path"])
            source_display_name = str(
                st.session_state.get("source_image_name", Path(query_path).name)
            )
            source_notice = f"Image source sélectionnée depuis les résultats : `{source_display_name}`"
        elif uploaded is not None:
            suffix = Path(uploaded.name).suffix if uploaded.name else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded.read())
                query_path = f.name
            source_display_name = uploaded.name or Path(query_path).name

        return QuerySource(
            query_path=query_path,
            source_display_name=source_display_name,
            source_notice=source_notice,
        )

    # ---------------------------------------------------------------------
    # SECTION 3 - PIPELINE MÉTIER : RETRIEVAL ET CONTEXTES PARTAGÉS
    # ---------------------------------------------------------------------
    def _build_results_context(self, source: QuerySource, result_count: int) -> ArtworkResultsContext:
        """
        Exécute le retrieval puis construit le contexte métier partagé.

        C'est ici que l'on bascule du "contexte d'entrée" vers le "contexte
        de comparaison" utilisé ensuite par presque toutes les sections.
        """
        assert self.retriever is not None
        assert source.query_path is not None

        with st.spinner("Recherche des œuvres similaires..."):
            results = self.retriever.top_k_similar(source.query_path, k=result_count)

        if not results:
            st.error("Aucun résultat n'a été retourné par le moteur de recherche.")
            st.stop()

        best = results[0]
        second_best = results[1] if len(results) > 1 else None
        third_best = results[2] if len(results) > 2 else None
        source_artist, source_title = _extract_artist_and_title(
            source.source_display_name or source.query_path
        )

        rows = self._build_result_rows(results)
        df_results = pd.DataFrame(rows)
        _update_internal_dataframe_from_results(df_results, source_artist, source_title)

        return ArtworkResultsContext(
            query_path=source.query_path,
            source_display_name=source.source_display_name,
            source_artist=source_artist,
            source_title=source_title,
            results=results,
            rows=rows,
            df_results=df_results,
            best=best,
            second_best=second_best,
            third_best=third_best,
        )

    def _build_result_rows(self, results: list[dict[str, object]]) -> list[dict[str, object]]:
        """
        Traduit les résultats bruts du moteur en lignes orientées interface.

        Le moteur expose une structure compacte ; l'UI a besoin au contraire de
        champs explicites et déjà formatés pour éviter de refaire les mêmes
        transformations à plusieurs endroits.
        """
        rows: list[dict[str, object]] = []
        for i, res in enumerate(results):
            artist, title = _extract_artist_and_title(res["filepath"])
            rows.append(
                {
                    "rang": i + 1,
                    "artiste": artist,
                    "tableau": title,
                    "style": res["style"],
                    "similarité": round(float(res["similarity"]), 4),
                    "fichier": Path(str(res["filepath"])).name,
                    "chemin": str(res["filepath"]),
                }
            )
        return rows

    # ---------------------------------------------------------------------
    # SECTION 4 - RENDU UI PRINCIPAL : SOURCE, IA, COMPARAISON, TABLEAU
    # ---------------------------------------------------------------------
    def _render_source_section(self, context: ArtworkResultsContext) -> None:
        """
        Rend le cartouche de l'image source et son style suggéré.
        """
        st.subheader("Image source")
        source_image_width = _compute_source_image_display_width(context.query_path)
        st.image(context.query_path, caption="Image requête", width=source_image_width)

        st.markdown(
            f"""
            <p style="line-height:1.1; margin:0;">
                <strong>{context.source_artist}</strong><br>
                <em>{context.source_title}</em>
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="museum-style-chip">
                Style suggéré • {context.best['style']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _build_ai_analysis_state(
        self,
        context: ArtworkResultsContext,
        config: AppConfig,
    ) -> AIAnalysisState:
        """
        Prépare et, si nécessaire, exécute la phase d'analyse IA.

        Cette méthode concentre :
        - l'activation UI ;
        - la préparation des oeuvres candidates ;
        - l'appel potentiellement coûteux à l'agent ;
        - la capture des erreurs sans interrompre le reste de l'app.
        """
        # REVIEW NOTE:
        # Le cache IA est indexé par :
        # - la liste d'oeuvres candidates sérialisée ;
        # - l'oeuvre d'intérêt ;
        # - le profil IA.
        # Cela évite un bug classique où un changement de profil réutiliserait
        # une ancienne réponse générée avec un autre prompt.
        show_ai_analyses = False
        if config.ai_agent_enabled:
            show_ai_analyses = st.checkbox(
                "Afficher les analyses IA sous les tableaux",
                value=False,
                key="show_ai_analyses",
            )

        artwork_of_interest = " ".join(
            part for part in [str(context.source_artist).strip(), str(context.source_title).strip()] if part
        )
        candidates_df = pd.DataFrame(
            [{"Artiste": row["artiste"], "Titre": row["tableau"], "Année": None} for row in context.rows]
        )

        payload: dict[str, str] | None = None
        error: str | None = None
        enabled = config.ai_agent_enabled and show_ai_analyses

        if enabled and not candidates_df.empty:
            candidates_json = json.dumps(candidates_df.to_dict(orient="records"), ensure_ascii=False)
            ai_progress_text = st.empty()
            ai_progress_bar = st.progress(0)
            try:
                ai_progress_text.caption("Chargement des analyses IA : préparation des œuvres")
                ai_progress_bar.progress(0.2)
                ai_progress_text.caption("Chargement des analyses IA : envoi de la requête")
                ai_progress_bar.progress(0.45)
                payload = _get_cached_ai_analysis(
                    candidates_json,
                    artwork_of_interest,
                    config.ai_profile_name,
                )
                ai_progress_text.caption("Chargement des analyses IA : extraction des chapitres")
                ai_progress_bar.progress(0.8)
                _extract_chapters_payload(payload["final_output"])
                ai_progress_text.caption("Chargement des analyses IA : terminé")
                ai_progress_bar.progress(1.0)
            except Exception as exc:
                error = str(exc)
            finally:
                ai_progress_text.empty()
                ai_progress_bar.empty()

        return AIAnalysisState(
            enabled=enabled,
            profile_name=config.ai_profile_name,
            payload=payload,
            error=error,
            artwork_of_interest=artwork_of_interest,
            candidates_df=candidates_df,
        )

    def _render_source_ai_section(self, context: ArtworkResultsContext, ai_state: AIAnalysisState) -> None:
        """
        Affiche l'analyse IA dédiée à l'oeuvre source.

        Important : ce bloc n'affiche rien tant que le mapping source -> chapitre
        n'a pas été résolu de façon fiable.
        """
        if not ai_state.enabled or ai_state.payload is None:
            return

        source_analysis = _match_source_artwork_analysis(
            ai_state.payload["final_output"],
            context.source_artist,
            context.source_title,
            ai_state.artwork_of_interest,
            context.source_display_name,
        )
        if not source_analysis:
            return

        with st.expander(
            f"Analyse IA ({ai_state.profile_name}) de l'image requête",
            expanded=False,
        ):
            st.markdown(
                f"""
                <div style="
                    margin-top: 0.2rem;
                    padding: 0.9rem 1rem;
                    border: 1px solid rgba(17, 17, 17, 0.1);
                    border-radius: 10px;
                    background: rgba(246, 243, 238, 0.72);
                    color: #1d1d1d;
                    line-height: 1.55;
                    font-size: 0.92rem;
                ">
                    {_analysis_text_to_html(source_analysis)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_visual_comparison(self, context: ArtworkResultsContext) -> None:
        """
        Affiche les cartes des oeuvres similaires et gère le "re-click" d'une
        oeuvre comme nouvelle source.
        """
        st.subheader("Comparaison visuelle")
        cols = st.columns(min(4, len(context.results)))

        for i, res in enumerate(context.results):
            artist, title = _extract_artist_and_title(res["filepath"])
            with cols[i % len(cols)]:
                st.image(
                    res["filepath"],
                    caption=f"Top {i + 1} — {res['style']} ({res['similarity']:.3f})",
                    width="stretch",
                )
                st.markdown(
                    f"""
                    <p style="line-height:1.1; margin:0;">
                        <strong>{artist}</strong>
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"*{title}*",
                    key=f"use-as-source-{i}",
                    help="Cliquer pour charger cette œuvre comme nouvelle image source",
                ):
                    st.session_state["reset_gradcam_history"] = True
                    st.session_state["reset_ai_analyses"] = True
                    st.session_state["source_mode"] = "gallery"
                    st.session_state["source_image_path"] = str(res["filepath"])
                    st.session_state["source_image_name"] = Path(str(res["filepath"])).name
                    st.rerun()

    def _render_global_ai_section(
        self,
        context: ArtworkResultsContext,
        ai_state: AIAnalysisState,
    ) -> None:
        """
        Rend le panneau d'analyses IA des résultats et la synthèse globale.

        Cette section consomme le même payload IA que le cartouche source mais
        avec un mapping différent : ici on redistribue l'analyse par oeuvre
        candidate, puis on extrait la conclusion globale.
        """
        if not ai_state.enabled:
            return

        with st.expander(
            f"Analyses IA ({ai_state.profile_name}) et comparaison stylistique globale",
            expanded=False,
        ):
            if ai_state.payload is not None:
                analysis_cols = st.columns(min(4, len(context.rows)))
                for i, row in enumerate(context.rows):
                    artwork_analysis = _match_artwork_analysis(
                        ai_state.payload["final_output"],
                        row["artiste"],
                        row["tableau"],
                        i,
                    )
                    with analysis_cols[i % len(analysis_cols)]:
                        st.markdown(f"**{row['artiste']}**  \n*{row['tableau']}*")
                        if artwork_analysis:
                            st.markdown(
                                f"""
                                <div style="
                                    margin-top: 0.2rem;
                                    margin-bottom: 0.9rem;
                                    padding: 0.9rem 1rem;
                                    border: 1px solid rgba(17, 17, 17, 0.08);
                                    border-radius: 10px;
                                    background: rgba(255, 255, 255, 0.55);
                                    color: #1d1d1d;
                                    line-height: 1.55;
                                    font-size: 0.92rem;
                                ">
                                    {_analysis_text_to_html(artwork_analysis)}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("Analyse IA non trouvée pour cette œuvre.")

                st.markdown("---")
                st.markdown("**Comparaison stylistique globale**")
                global_analysis = _extract_global_analysis(ai_state.payload["final_output"])
                if global_analysis:
                    st.markdown(
                        f"""
                        <div style="
                            margin-top: 0.2rem;
                            padding: 1rem 1.1rem;
                            border-left: 6px solid #b1221c;
                            border-radius: 10px;
                            background: rgba(255, 255, 255, 0.6);
                            color: #1d1d1d;
                            line-height: 1.6;
                        ">
                            {_analysis_text_to_html(global_analysis)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("La comparaison stylistique globale n'a pas été trouvée dans la réponse IA.")
            elif ai_state.error is not None:
                st.warning(f"Analyse IA indisponible : {ai_state.error}")

    def _render_summary(self, df_results: pd.DataFrame) -> None:
        """
        Rend le tableau récapitulatif brut des résultats du retrieval.
        """
        st.subheader("Résumé des résultats")
        st.dataframe(df_results, width="stretch", hide_index=True)

    # ---------------------------------------------------------------------
    # SECTION 5 - VISUALISATIONS : UMAP ET REPÈRES VISUELS
    # ---------------------------------------------------------------------
    def _render_umap(self, context: ArtworkResultsContext) -> None:
        """
        Rend la visualisation UMAP interactive si le bundle latent est disponible.

        Le rôle de cette méthode est surtout de préparer un DataFrame orienté
        visualisation puis de déléguer le marquage des points importants
        (upload, top-1, top-2, top-3).
        """
        if self.latent_error is not None:
            st.warning(f"Visualisation UMAP indisponible : {self.latent_error}")
            return

        if self.latent_bundle is None:
            return

        with st.expander("Espace latent (UMAP interactif)", expanded=False):
            latent_2d, labels, classnames, filenames = self.latent_bundle
            style_names = _build_style_names(labels, classnames)
            short_filenames = [Path(str(fp)).name for fp in filenames]
            artists = []
            titles = []

            for fp in filenames:
                artist, title = _extract_artist_and_title(str(fp))
                artists.append(artist)
                titles.append(title)

            similarity_by_filepath = {
                str(res["filepath"]): f"{float(res['similarity']) * 100:.1f}%"
                for res in context.results
            }

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

            df_umap = pd.DataFrame(
                {
                    "x": latent_2d[:, 0],
                    "y": latent_2d[:, 1],
                    "Label": labels,
                    "Style": style_names,
                    "Artiste": artists,
                    "Tableau": titles,
                    "Fichier": short_filenames,
                    "tooltip_html": tooltip_html,
                    "filepath": [str(fp) for fp in filenames],
                }
            )

            styles_disponibles = sorted(df_umap["Style"].astype(str).unique().tolist())
            styles_selectionnes = st.multiselect(
                "Filtrer les styles affichés dans l'UMAP",
                options=styles_disponibles,
                default=styles_disponibles,
            )

            df_umap_filtered = df_umap[df_umap["Style"].isin(styles_selectionnes)].copy()
            if df_umap_filtered.empty:
                st.warning("Aucun point à afficher : aucun style sélectionné.")
                return

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
            fig.update_traces(marker=dict(size=7), hovertemplate="%{customdata[0]}<extra></extra>")
            self._add_umap_highlights(fig, context, latent_2d, filenames)
            fig.update_layout(
                xaxis_title="Dimension UMAP 1",
                yaxis_title="Dimension UMAP 2",
                legend_title="Styles",
                height=700,
                hoverlabel=dict(font=dict(size=16)),
            )

            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Chaque point représente une œuvre projetée dans un espace latent 2D. "
                "Les couleurs correspondent aux styles artistiques. "
                "Les points annotés correspondent au tableau uploadé ainsi qu'aux meilleurs résultats top-1, top-2 et top-3."
            )
            st.caption(
                "Note : la proximité visuelle dans l'UMAP ne correspond pas toujours exactement "
                "au classement de similarité, car la projection 2D déforme partiellement "
                "les distances calculées dans l'espace latent d'origine."
            )

    def _add_umap_highlights(self, fig, context: ArtworkResultsContext, latent_2d, filenames) -> None:
        """
        Ajoute les repères métier sur le scatter UMAP.

        L'upload n'existe pas directement dans la projection ; sa position est
        estimée à partir des meilleurs voisins connus dans l'espace latent.
        """
        # REVIEW NOTE:
        # Le point "Upload" n'est pas un embedding réellement projeté dans
        # l'UMAP chargé depuis disque. On construit donc une approximation
        # pondérée à partir des top voisins pour donner un repère visuel utile
        # sans prétendre à une position exacte du tableau source.
        upload_anchor_points = []
        for res in context.results[:3]:
            idx_res = _find_best_index(filenames, str(res["filepath"]))
            if idx_res is None:
                continue
            weight = max(float(res["similarity"]), 0.0)
            upload_anchor_points.append(
                (float(latent_2d[idx_res, 0]), float(latent_2d[idx_res, 1]), weight)
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
                        f"Artiste: {context.source_artist}<br>"
                        f"Tableau: {context.source_title}<br>"
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

        self._add_result_highlight(
            fig, latent_2d, filenames, context.best, "Top 1", "Top-1 sélectionné",
            "top center", "circle-open", "black", 18, 20, -30, "white"
        )
        if context.second_best is not None:
            self._add_result_highlight(
                fig, latent_2d, filenames, context.second_best, "Top 2", "Top-2 sélectionné",
                "bottom center", "diamond-open", "#b54708", 16, -24, 34, "#fff7ed"
            )
        if context.third_best is not None:
            self._add_result_highlight(
                fig, latent_2d, filenames, context.third_best, "Top 3", "Top-3 sélectionné",
                "middle right", "square-open", "#175cd3", 15, 34, 18, "#eff8ff"
            )

    def _add_result_highlight(
        self,
        fig,
        latent_2d,
        filenames,
        result: dict[str, object],
        annotation_text: str,
        trace_name: str,
        text_position: str,
        symbol: str,
        line_color: str,
        marker_size: int,
        ax: int,
        ay: int,
        annotation_bg: str,
    ) -> None:
        """
        Factorise l'ajout d'un repère Plotly pour un résultat important.

        Cette extraction évite de dupliquer trois fois la même logique pour
        top-1 / top-2 / top-3, ce qui facilite les futurs ajustements visuels.
        """
        idx_result = _find_best_index(filenames, str(result["filepath"]))
        if idx_result is None:
            return

        x_value = latent_2d[idx_result, 0]
        y_value = latent_2d[idx_result, 1]
        filename = Path(str(result["filepath"])).name
        artist, title = _extract_artist_and_title(result["filepath"])

        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[y_value],
                mode="markers+text",
                name=trace_name,
                text=[result["style"]],
                textposition=text_position,
                marker=dict(size=marker_size, symbol=symbol, line=dict(width=3, color=line_color)),
                hovertemplate=(
                    f"<b>{trace_name}</b><br>"
                    f"Style: {result['style']}<br>"
                    f"Artiste: {artist}<br>"
                    f"Tableau: {title}<br>"
                    f"Fichier: {filename}<br>"
                    f"Similarité: {float(result['similarity']) * 100:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=x_value,
            y=y_value,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            ax=ax,
            ay=ay,
            bgcolor=annotation_bg,
            bordercolor=line_color,
            borderwidth=1,
        )

    # ---------------------------------------------------------------------
    # SECTION 6 - VISUALISATION EXPLICATIVE : GRAD-CAM
    # ---------------------------------------------------------------------
    def _render_gradcam(self, context: ArtworkResultsContext, gradcam_pair_count: int) -> None:
        """
        Point d'entrée UI de la section Grad-CAM.

        Cette méthode décide seulement si l'explication visuelle peut être
        lancée. Le calcul effectif est délégué à `_render_gradcam_history`.
        """
        show_gradcam_history = st.checkbox("Grad-CAM history", key="show_gradcam_history")

        available_layers: list[str] = []
        if show_gradcam_history and self.retriever is not None:
            available_layers = self.retriever.available_explanation_layers()
            if not available_layers:
                st.warning("Aucune couche compatible n'a été trouvée pour Grad-CAM++.")

        if show_gradcam_history and available_layers:
            self._render_gradcam_history(context, available_layers, gradcam_pair_count)
        else:
            st.info(
                "Active l'option Grad-CAM++ pour visualiser les zones qui "
                "contribuent à la similarité du top-1."
            )

    def _render_gradcam_history(
        self,
        context: ArtworkResultsContext,
        available_layers: list[str],
        gradcam_pair_count: int,
    ) -> None:
        """
        Exécute et rend l'historique des cartes Grad-CAM++.

        Les couches sont échantillonnées sur l'intervalle du réseau afin de
        montrer différentes profondeurs de représentation, plutôt qu'un seul
        niveau d'explication.
        """
        # REVIEW NOTE:
        # Le choix des couches n'est pas une simple liste fixe :
        # on répartit aléatoirement les indices sur l'intervalle 1..245 pour
        # varier la lecture d'une exécution à l'autre tout en couvrant le réseau
        # de manière relativement homogène.
        assert self.retriever is not None

        try:
            history_layer_numbers = _build_random_gradcam_layer_numbers(
                gradcam_pair_count,
                min_layer=1,
                max_layer=245,
            )
            history_layers, missing_history_layers = _select_explanation_layers(
                available_layers,
                history_layer_numbers,
            )
            _, layer_labels = _format_explanation_layer_options(available_layers)

            if not history_layers:
                st.warning("Aucune des couches de l'historique n'est disponible pour ce modèle.")
                st.stop()

            with st.spinner("Calcul des cartes Grad-CAM++..."):
                history_explanations = []
                history_total = len(history_layers)
                history_progress_text = st.empty()
                history_progress_bar = st.progress(0)

                for history_index, (layer_number, layer_name) in enumerate(history_layers, start=1):
                    history_progress_text.caption(
                        f"Chargement de Grad-CAM history : {history_index}/{history_total} "
                        f"(couche {layer_number})"
                    )
                    history_explanations.append(
                        (
                            layer_number,
                            self.retriever.explain_similarity(
                                context.query_path,
                                context.best["filepath"],
                                target_layer_name=layer_name,
                            ),
                        )
                    )
                    history_progress_bar.progress(history_index / history_total)

                history_progress_text.empty()
                history_progress_bar.empty()

            best_artist, best_title = _extract_artist_and_title(context.best["filepath"])
            with st.expander("Explication visuelle (Grad-CAM++)", expanded=False):
                st.markdown(
                    f"""
                    <div class="museum-card">
                        <strong>Top-1 sélectionné</strong>
                        <em>{best_artist} • {best_title}</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("**Grad-CAM history**")
                st.caption(
                    f"{len(history_explanations)} paires de cartes affichees pour les couches "
                    f"{', '.join(str(layer_number) for layer_number in history_layer_numbers)}."
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

    # ---------------------------------------------------------------------
    # SECTION 7 - ÉTATS DE GARDE / FALLBACKS D'ACCUEIL
    # ---------------------------------------------------------------------
    def _render_unavailable_state(self, uploaded) -> None:
        """
        Gère les états d'accueil / indisponibilité quand le moteur n'est pas prêt.
        """
        upload_disabled = bool(
            (self.retriever_error is not None)
            or (not bool(self.runtime_status and self.runtime_status["upload_enabled"]))
        )
        if uploaded is None:
            if upload_disabled:
                st.info("Upload désactivé tant que le modèle, les embeddings et `data/out` ne sont pas synchronisés.")
            else:
                st.info("Charge une image pour lancer la recherche.")
            return

        st.error("Le moteur n'est pas prêt. Vérifie le modèle, les embeddings et data/out.")


ArtXplainApp().run()
