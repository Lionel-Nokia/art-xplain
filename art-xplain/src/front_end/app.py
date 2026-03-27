from __future__ import annotations

import atexit
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.front_end.parsing import (  # noqa: E402
    analysis_text_to_html,
    extract_chapters_payload,
    extract_global_analysis,
    match_artwork_analysis,
    match_source_artwork_analysis,
)
from src.front_end.ui import (  # noqa: E402
    compute_source_image_display_width,
    render_runtime_status,
    setup_page,
)
from src.front_end.visualization import (  # noqa: E402
    build_random_gradcam_layer_numbers,
    build_style_names,
    extract_artist_and_title,
    find_best_index,
    format_explanation_layer_options,
    get_retriever,
    load_latent_and_meta,
    select_explanation_layers,
)
from src.ia_agent import run_analysis  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


ASSETS_DIR = PROJECT_ROOT / "assets"
APP_LOGO_PATH = ASSETS_DIR / "artxplain-logo.svg"
INTERNAL_DF_DIR = PROJECT_ROOT / "data"
INTERNAL_DF_PATH = INTERNAL_DF_DIR / "internal_artworks.csv"
INTERNAL_DF_COLUMNS = ["artiste", "tableau", "style", "fichier", "analyse", "similarite"]
AI_GUIDE_PROFILE_NAME = "guide_musée"


class InternalArtworkStore:
    """
    Encapsule le mini stockage CSV utilisé par l'app Streamlit.

    Cette classe regroupe tout ce qui concerne :
    - la normalisation du schéma,
    - la lecture / écriture disque,
    - l'initialisation du `session_state`,
    - la mise à jour de l'historique des similarités.

    En revue de code, cela évite d'avoir des helpers de persistance dispersés
    au milieu des méthodes de rendu d'interface.
    """

    SESSION_KEY = "internal_artworks_df"
    SHUTDOWN_HOOK_KEY = "_internal_dataframe_shutdown_hook_registered"

    @staticmethod
    def normalize_similarity_json(value: object) -> str:
        # Le CSV peut contenir des listes, des chaînes vides, des NaN ou des
        # valeurs partiellement valides. On ramène toujours ce champ à un JSON
        # texte représentant une liste pour garder une lecture robuste.
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
        return json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, list) else "[]"

    @classmethod
    def parse_similarity_json(cls, value: object) -> list[dict[str, object]]:
        try:
            parsed = json.loads(cls.normalize_similarity_json(value))
        except json.JSONDecodeError:
            return []
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    @staticmethod
    def is_unknown_metadata(value: object) -> bool:
        # L'app utilise "Inconnu" comme valeur de repli quand le parsing du nom
        # de fichier ne permet pas d'identifier correctement l'œuvre.
        return str(value).strip().lower() == "inconnu"

    @staticmethod
    def empty_dataframe() -> pd.DataFrame:
        return pd.DataFrame(columns=INTERNAL_DF_COLUMNS)

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Le CSV interne joue le rôle d'un stockage applicatif léger.
        # On réimpose donc son schéma à chaque lecture et écriture pour éviter
        # les dérives de colonnes ou les types incohérents Pandas.
        normalized = df.copy()
        for column in INTERNAL_DF_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = ""
        normalized = normalized[INTERNAL_DF_COLUMNS].fillna("")
        normalized["similarite"] = normalized["similarite"].apply(cls.normalize_similarity_json)
        return normalized.astype(str)

    @classmethod
    def load(cls) -> pd.DataFrame:
        if not INTERNAL_DF_PATH.exists():
            return cls.empty_dataframe()
        try:
            return cls.normalize_dataframe(pd.read_csv(INTERNAL_DF_PATH))
        except Exception:
            return cls.empty_dataframe()

    @classmethod
    def save(cls, df: pd.DataFrame) -> None:
        ensure_dir(INTERNAL_DF_DIR)
        cls.normalize_dataframe(df).to_csv(INTERNAL_DF_PATH, index=False)

    @classmethod
    def register_shutdown_hook(cls) -> None:
        # Streamlit rerun le script, mais le processus peut survivre plusieurs
        # interactions. Ce hook assure une persistance "best effort" à la sortie.
        if st.session_state.get(cls.SHUTDOWN_HOOK_KEY):
            return

        def persist_on_exit() -> None:
            df = st.session_state.get(cls.SESSION_KEY)
            if isinstance(df, pd.DataFrame):
                try:
                    cls.save(df)
                except Exception:
                    pass

        atexit.register(persist_on_exit)
        st.session_state[cls.SHUTDOWN_HOOK_KEY] = True

    @classmethod
    def initialize_session_state(cls) -> None:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = cls.load()
        cls.register_shutdown_hook()

    @classmethod
    def update_from_results(
        cls,
        df_results: pd.DataFrame,
        source_artist: str,
        source_title: str,
    ) -> pd.DataFrame:
        # Cette routine enrichit l'historique local des œuvres rencontrées.
        # Elle combine trois responsabilités métier :
        # - dédupliquer les œuvres déjà connues,
        # - mémoriser de nouvelles œuvres rencontrées dans les résultats,
        # - historiser quels couples source -> candidate ont déjà été vus.
        internal_df = cls.normalize_dataframe(st.session_state[cls.SESSION_KEY])
        if cls.is_unknown_metadata(source_artist) or cls.is_unknown_metadata(source_title):
            st.session_state[cls.SESSION_KEY] = internal_df
            return internal_df

        source_df = (
            df_results.loc[:, ["artiste", "tableau", "style", "fichier", "similarité"]]
            .copy()
            .fillna("")
            .astype(str)
        )
        source_df = source_df[
            ~source_df["artiste"].apply(cls.is_unknown_metadata)
            & ~source_df["tableau"].apply(cls.is_unknown_metadata)
        ].copy()
        source_df["analyse"] = ""
        source_df["similarite"] = "[]"
        source_df = source_df.rename(columns={"similarité": "similarite_courante"})
        if source_df.empty:
            st.session_state[cls.SESSION_KEY] = internal_df
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
        # `existing_pairs` sert à éviter les doublons "logiques" d'une œuvre.
        # `existing_keys` garde un niveau plus précis quand plusieurs fichiers
        # pourraient représenter une œuvre proche ou identique.
        rows_to_add = source_df[~source_df["pair_key"].isin(existing_pairs)].copy()
        if not rows_to_add.empty:
            internal_df = pd.concat([internal_df, rows_to_add[INTERNAL_DF_COLUMNS]], ignore_index=True)

        results_by_key = {
            (
                str(row["artiste"]).strip(),
                str(row["tableau"]).strip(),
                str(row["fichier"]).strip(),
            ): float(row["similarite_courante"])
            for _, row in source_df.iterrows()
        }
        source_signature = (str(source_artist).strip(), str(source_title).strip())

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

            history = cls.parse_similarity_json(row["similarite"])
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

        internal_df = cls.normalize_dataframe(internal_df)
        st.session_state[cls.SESSION_KEY] = internal_df
        cls.save(internal_df)
        return internal_df


class AIConfigService:
    """
    Centralise la lecture de la configuration de l'agent IA.

    L'objectif est de séparer :
    - la lecture du fichier YAML,
    - les valeurs par défaut métier,
    - l'exposition de paramètres prêts pour l'UI.
    """

    CONFIG_PATH = "config/config_agent.yaml"

    @classmethod
    def load_agent_config(cls) -> dict[str, object]:
        return load_config(cls.CONFIG_PATH)

    @classmethod
    def resolve_enabled(cls) -> bool:
        try:
            config = cls.load_agent_config()
        except Exception:
            return False
        return bool(config.get("ai-agent", {}).get("ai_active", True))

    @classmethod
    def resolve_default_profile_name(cls) -> str:
        try:
            config = cls.load_agent_config()
        except Exception:
            return AI_GUIDE_PROFILE_NAME
        profile_name = str(config.get("ai-agent", {}).get("default_profile", "")).strip()
        return profile_name or AI_GUIDE_PROFILE_NAME

    @classmethod
    def resolve_available_profile_names(cls) -> list[str]:
        try:
            config = cls.load_agent_config()
        except Exception:
            return [AI_GUIDE_PROFILE_NAME]
        profiles = config.get("ai-agent", {}).get("profiles", {})
        if isinstance(profiles, dict):
            names = [str(name).strip() for name in profiles if str(name).strip()]
            if names:
                return names
        return [AI_GUIDE_PROFILE_NAME]


class AIAnalysisService:
    """
    Fait le pont entre l'agent IA asynchrone et le flux synchrone Streamlit.

    Cette classe ne s'occupe pas du rendu. Elle fournit seulement :
    - le lancement de l'analyse,
    - la sérialisation / désérialisation des candidats,
    - un point d'entrée cacheable pour éviter les recalculs inutiles.
    """

    @staticmethod
    def run_async_analysis_sync(
        candidates_df: pd.DataFrame,
        artwork_of_interest: str,
        profile_name: str,
    ) -> dict[str, str]:
        def build_coroutine():
            return run_analysis(
                df=candidates_df,
                artwork_of_interest=artwork_of_interest,
                config_path=str(PROJECT_ROOT / "config" / "config_agent.yaml"),
                profile_name=profile_name,
                output_folder=str(PROJECT_ROOT / "outputs"),
                save_to_file=False,
            )

        # Streamlit vit dans un environnement synchrone, alors que l'agent IA
        # expose une coroutine. On gère à la fois le cas standard et le cas où
        # une boucle asyncio existe déjà dans le contexte d'exécution.
        try:
            analysis = asyncio.run(build_coroutine())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                analysis = loop.run_until_complete(build_coroutine())
            finally:
                loop.close()

        return {
            "final_output": str(analysis["result"].final_output).strip(),
            "output_path": str(analysis.get("output_path", "")),
        }

    @staticmethod
    def deserialize_candidates(candidates_json: str) -> pd.DataFrame:
        return pd.DataFrame(json.loads(candidates_json))


class ResultsPresenter:
    """
    Prépare les structures de données partagées entre retrieval, IA et UI.

    Le rôle de cette classe est de produire des objets "présentation"
    suffisamment riches pour le front sans lui faire manipuler directement
    le format brut du retriever partout.
    """

    @staticmethod
    def build_result_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, result in enumerate(results):
            artist, title = extract_artist_and_title(result["filepath"])
            rows.append(
                {
                    "rang": index + 1,
                    "artiste": artist,
                    "tableau": title,
                    "style": result["style"],
                    "similarité": round(float(result["similarity"]), 4),
                    "fichier": Path(str(result["filepath"])).name,
                    "chemin": str(result["filepath"]),
                }
            )
        return rows

    @staticmethod
    def build_candidates_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"Artiste": row["artiste"], "Titre": row["tableau"], "Année": None} for row in rows]
        )

    @staticmethod
    def build_similarity_lookup(results: list[dict[str, object]]) -> dict[str, str]:
        return {
            str(result["filepath"]): f"{float(result['similarity']) * 100:.1f}%"
            for result in results
        }


class UmapPlotBuilder:
    """
    Regroupe les helpers de construction et d'enrichissement de la figure UMAP.

    On isole ce code car il porte une logique de visualisation spécifique :
    dataframe Plotly, tooltips enrichis et points de surbrillance.
    """

    @staticmethod
    def build_dataframe(
        latent_2d,
        labels,
        classnames,
        filenames,
        results: list[dict[str, object]],
    ) -> pd.DataFrame:
        style_names = build_style_names(labels, classnames)
        artists, titles = zip(*(extract_artist_and_title(str(filepath)) for filepath in filenames))
        similarity_by_filepath = ResultsPresenter.build_similarity_lookup(results)

        tooltip_html = []
        for filepath, style, artist, title in zip(filenames, style_names, artists, titles):
            lines = [
                f"<b>Style:</b> {style}",
                f"<b>Artiste:</b> {artist}",
                f"<b>Tableau:</b> {title}",
            ]
            similarity = similarity_by_filepath.get(str(filepath))
            if similarity:
                lines.append(f"<b>Similarité:</b> {similarity}")
            tooltip_html.append("<br>".join(lines))

        return pd.DataFrame(
            {
                "x": latent_2d[:, 0],
                "y": latent_2d[:, 1],
                "Label": labels,
                "Style": style_names,
                "Artiste": artists,
                "Tableau": titles,
                "Fichier": [Path(str(filepath)).name for filepath in filenames],
                "tooltip_html": tooltip_html,
                "filepath": [str(filepath) for filepath in filenames],
            }
        )

    @staticmethod
    def add_highlights(fig, context: "ArtworkResultsContext", latent_2d, filenames) -> None:
        # L'image uploadée n'a pas sa propre projection UMAP à l'écran.
        # On l'approxime donc par barycentre pondéré de ses meilleurs voisins,
        # ce qui fournit un repère visuel sans recalculer d'UMAP à la volée.
        upload_anchor_points = []
        for result in context.results[:3]:
            result_index = find_best_index(filenames, str(result["filepath"]))
            if result_index is not None:
                weight = max(float(result["similarity"]), 0.0)
                upload_anchor_points.append(
                    (
                        float(latent_2d[result_index, 0]),
                        float(latent_2d[result_index, 1]),
                        weight,
                    )
                )

        if upload_anchor_points:
            total_weight = sum(weight for _, _, weight in upload_anchor_points) or float(
                len(upload_anchor_points)
            )
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
                    marker=dict(size=17, symbol="star", color="#d92d20", line=dict(width=1.5, color="white")),
                    showlegend=False,
                )
            )

        UmapPlotBuilder.add_result_highlight(
            fig,
            latent_2d,
            filenames,
            context.best,
            "Top 1",
            "Top-1 sélectionné",
            "top center",
            "circle-open",
            "black",
            18,
            20,
            -30,
            "white",
        )
        if context.second_best is not None:
            UmapPlotBuilder.add_result_highlight(
                fig,
                latent_2d,
                filenames,
                context.second_best,
                "Top 2",
                "Top-2 sélectionné",
                "bottom center",
                "diamond-open",
                "#b54708",
                16,
                -24,
                34,
                "#fff7ed",
            )
        if context.third_best is not None:
            UmapPlotBuilder.add_result_highlight(
                fig,
                latent_2d,
                filenames,
                context.third_best,
                "Top 3",
                "Top-3 sélectionné",
                "middle right",
                "square-open",
                "#175cd3",
                15,
                34,
                18,
                "#eff8ff",
            )

    @staticmethod
    def add_result_highlight(
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
        result_index = find_best_index(filenames, str(result["filepath"]))
        if result_index is None:
            return

        x_value = latent_2d[result_index, 0]
        y_value = latent_2d[result_index, 1]
        artist, title = extract_artist_and_title(result["filepath"])
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
                    f"Similarité: {float(result['similarity']) * 100:.1f}%<extra></extra>"
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


@st.cache_data(show_spinner=False)
def is_ai_agent_enabled() -> bool:
    return AIConfigService.resolve_enabled()


@st.cache_data(show_spinner=False)
def get_default_ai_profile_name() -> str:
    return AIConfigService.resolve_default_profile_name()


@st.cache_data(show_spinner=False)
def get_available_ai_profile_names() -> list[str]:
    return AIConfigService.resolve_available_profile_names()


@st.cache_data(show_spinner=False)
def get_cached_ai_analysis(
    candidates_json: str,
    artwork_of_interest: str,
    profile_name: str,
) -> dict[str, str]:
    candidates_df = AIAnalysisService.deserialize_candidates(candidates_json)
    return AIAnalysisService.run_async_analysis_sync(candidates_df, artwork_of_interest, profile_name)


@dataclass
class AppConfig:
    result_count: int
    ai_profile_name: str
    gradcam_pair_count: int
    ai_agent_enabled: bool


@dataclass
class QuerySource:
    query_path: str | None
    source_display_name: str | None
    source_notice: str | None


@dataclass
class ArtworkResultsContext:
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
    enabled: bool
    profile_name: str
    payload: dict[str, str] | None
    error: str | None
    artwork_of_interest: str
    candidates_df: pd.DataFrame


class ArtXplainApp:
    """
    Orchestrateur principal de l'écran Streamlit.

    Les helpers métier ont été déplacés dans des classes dédiées pour que
    cette classe reste centrée sur le flux utilisateur :
    - chargement des ressources,
    - résolution de la requête,
    - rendu progressif des sections,
    - pilotage du `session_state`.
    """

    def __init__(self) -> None:
        self.retriever = None
        self.retriever_error = None
        self.latent_bundle = None
        self.latent_error = None
        self.runtime_status = None
        self.available_ai_profiles: list[str] = []
        self.default_ai_profile_name = AI_GUIDE_PROFILE_NAME

    def run(self) -> None:
        # Ordre d'exécution important :
        # 1. charger les ressources,
        # 2. initialiser l'état Streamlit,
        # 3. résoudre la source image,
        # 4. calculer les résultats,
        # 5. dérouler les sections dépendantes de ce contexte.
        self._load_resources()
        self.runtime_status = render_runtime_status()
        self.available_ai_profiles = get_available_ai_profile_names()
        self.default_ai_profile_name = get_default_ai_profile_name()
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

        context = self._build_results_context(source, config.result_count)
        self._render_source_section(context)
        ai_state = self._build_ai_analysis_state(context, config)
        self._render_source_ai_section(context, ai_state)
        self._render_visual_comparison(context)
        self._render_global_ai_section(context, ai_state)
        self._render_summary(context.df_results)
        self._render_umap(context)
        self._render_gradcam(context, config.gradcam_pair_count)

    def _load_resources(self) -> None:
        try:
            self.retriever = get_retriever()
        except Exception as exc:
            self.retriever_error = exc
        try:
            self.latent_bundle = load_latent_and_meta()
        except Exception as exc:
            self.latent_error = exc

    def _initialize_session_state(self) -> None:
        # On garde ici uniquement les flags d'interface transitoires.
        # Le stockage persistant des œuvres internes est délégué à
        # `InternalArtworkStore` pour éviter de mélanger UI et persistance.
        if "show_gradcam_history" not in st.session_state:
            st.session_state["show_gradcam_history"] = False
        if st.session_state.pop("reset_gradcam_history", False):
            st.session_state["show_gradcam_history"] = False
        if st.session_state.pop("reset_ai_analyses", False):
            st.session_state["show_ai_analyses"] = False

    def _render_uploader(self):
        # Changer d'image doit invalider les explications précédentes,
        # sinon l'utilisateur pourrait visualiser des analyses produites
        # pour une requête antérieure.
        upload_disabled = bool(
            (self.retriever_error is not None)
            or (not bool(self.runtime_status and self.runtime_status["upload_enabled"]))
        )
        uploaded = st.file_uploader(
            "Upload une image (jpg/png/webp)",
            type=["jpg", "jpeg", "png", "webp"],
            disabled=upload_disabled,
        )
        uploaded_signature = f"{uploaded.name}:{uploaded.size}" if uploaded is not None else None
        if uploaded_signature != st.session_state.get("uploaded_signature"):
            st.session_state["uploaded_signature"] = uploaded_signature
            st.session_state["show_gradcam_history"] = False
            st.session_state["show_ai_analyses"] = False
            if uploaded_signature is not None:
                st.session_state["source_mode"] = "uploaded"
        return uploaded

    def _render_configuration_panel(self) -> AppConfig:
        ai_agent_enabled = is_ai_agent_enabled()
        with st.expander("Configuration", expanded=False):
            selected_result_count = st.slider("Nombre de tableaux comparés", 1, 4, 4, key="result_count")
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
            )
            gradcam_pair_count = st.slider("Nombre de paires Grad-CAM", 1, 30, 10, key="gradcam_pair_count")
            if not ai_agent_enabled:
                st.caption("L'agent IA est désactivé dans la configuration actuelle.")
        return AppConfig(
            int(selected_result_count),
            str(selected_ai_profile).strip() or self.default_ai_profile_name,
            int(gradcam_pair_count),
            ai_agent_enabled,
        )

    def _resolve_query_source(self, uploaded) -> QuerySource:
        # Deux modes source coexistent :
        # - un fichier uploadé par l'utilisateur,
        # - une œuvre déjà affichée, rechargée comme nouvelle source.
        if st.session_state.get("source_mode") == "gallery" and st.session_state.get("source_image_path"):
            query_path = str(st.session_state["source_image_path"])
            source_display_name = str(
                st.session_state.get("source_image_name", Path(query_path).name)
            )
            return QuerySource(
                query_path,
                source_display_name,
                f"Image source sélectionnée depuis les résultats : `{source_display_name}`",
            )

        if uploaded is not None:
            suffix = Path(uploaded.name).suffix if uploaded.name else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded.read())
                query_path = temp_file.name
            return QuerySource(query_path, uploaded.name or Path(query_path).name, None)

        return QuerySource(None, None, None)

    def _build_results_context(self, source: QuerySource, result_count: int) -> ArtworkResultsContext:
        with st.spinner("Recherche des œuvres similaires..."):
            results = self.retriever.top_k_similar(source.query_path, k=result_count)
        if not results:
            st.error("Aucun résultat n'a été retourné par le moteur de recherche.")
            st.stop()

        rows = ResultsPresenter.build_result_rows(results)
        df_results = pd.DataFrame(rows)
        source_artist, source_title = extract_artist_and_title(source.source_display_name or source.query_path)
        InternalArtworkStore.update_from_results(df_results, source_artist, source_title)
        return ArtworkResultsContext(
            source.query_path,
            source.source_display_name,
            source_artist,
            source_title,
            results,
            rows,
            df_results,
            results[0],
            results[1] if len(results) > 1 else None,
            results[2] if len(results) > 2 else None,
        )

    def _render_source_section(self, context: ArtworkResultsContext) -> None:
        st.subheader("Image source")
        st.image(
            context.query_path,
            caption="Image requête",
            width=compute_source_image_display_width(context.query_path),
        )
        st.markdown(
            (
                f"<p style='line-height:1.1; margin:0;'><strong>{context.source_artist}</strong>"
                f"<br><em>{context.source_title}</em></p>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='museum-style-chip'>Style suggéré • {context.best['style']}</div>",
            unsafe_allow_html=True,
        )

    def _build_ai_analysis_state(self, context: ArtworkResultsContext, config: AppConfig) -> AIAnalysisState:
        # Le déclenchement IA reste opt-in car c'est la partie la plus coûteuse
        # et la plus variable de l'app.
        show_ai_analyses = (
            st.checkbox("Afficher les analyses IA sous les tableaux", value=False, key="show_ai_analyses")
            if config.ai_agent_enabled
            else False
        )
        artwork_of_interest = " ".join(
            part
            for part in [str(context.source_artist).strip(), str(context.source_title).strip()]
            if part
        )
        candidates_df = ResultsPresenter.build_candidates_dataframe(context.rows)
        enabled = config.ai_agent_enabled and show_ai_analyses
        payload, error = None, None

        if enabled and not candidates_df.empty:
            candidates_json = json.dumps(candidates_df.to_dict(orient="records"), ensure_ascii=False)
            try:
                payload = get_cached_ai_analysis(candidates_json, artwork_of_interest, config.ai_profile_name)
                extract_chapters_payload(payload["final_output"])
                # Cet appel sert aussi de validation implicite : si la structure
                # dérive trop côté agent, on le détecte avant plusieurs rendus UI.
            except Exception as exc:
                error = str(exc)

        return AIAnalysisState(
            enabled,
            config.ai_profile_name,
            payload,
            error,
            artwork_of_interest,
            candidates_df,
        )

    def _render_source_ai_section(self, context: ArtworkResultsContext, ai_state: AIAnalysisState) -> None:
        if not ai_state.enabled or ai_state.payload is None:
            return

        source_analysis = match_source_artwork_analysis(
            ai_state.payload["final_output"],
            context.source_artist,
            context.source_title,
            ai_state.artwork_of_interest,
            context.source_display_name,
        )
        if source_analysis:
            with st.expander(f"Analyse IA ({ai_state.profile_name}) de l'image requête", expanded=False):
                st.markdown(
                    (
                        "<div style='margin-top:0.2rem; padding:0.9rem 1rem; "
                        "border:1px solid rgba(17,17,17,0.1); border-radius:10px; "
                        "background:rgba(246,243,238,0.72); color:#1d1d1d; "
                        f"line-height:1.55; font-size:0.92rem;'>{analysis_text_to_html(source_analysis)}</div>"
                    ),
                    unsafe_allow_html=True,
                )

    def _render_visual_comparison(self, context: ArtworkResultsContext) -> None:
        st.subheader("Comparaison visuelle")
        cols = st.columns(min(4, len(context.results)))
        for index, result in enumerate(context.results):
            artist, title = extract_artist_and_title(result["filepath"])
            with cols[index % len(cols)]:
                st.image(
                    result["filepath"],
                    caption=f"Top {index + 1} — {result['style']} ({result['similarity']:.3f})",
                    width="stretch",
                )
                # On continue à reconstruire les métadonnées depuis le nom de fichier,
                # ce qui garde l'affichage stable même si le retriever expose peu de
                # champs descriptifs aujourd'hui.
                st.markdown(
                    f"<p style='line-height:1.1; margin:0;'><strong>{artist}</strong></p>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"*{title}*",
                    key=f"use-as-source-{index}",
                    help="Cliquer pour charger cette œuvre comme nouvelle image source",
                ):
                    st.session_state["reset_gradcam_history"] = True
                    st.session_state["reset_ai_analyses"] = True
                    st.session_state["source_mode"] = "gallery"
                    st.session_state["source_image_path"] = str(result["filepath"])
                    st.session_state["source_image_name"] = Path(str(result["filepath"])).name
                    st.rerun()

    def _render_global_ai_section(self, context: ArtworkResultsContext, ai_state: AIAnalysisState) -> None:
        if not ai_state.enabled:
            return

        with st.expander(
            f"Analyses IA ({ai_state.profile_name}) et comparaison stylistique globale",
            expanded=False,
        ):
            if ai_state.payload is not None:
                analysis_cols = st.columns(min(4, len(context.rows)))
                for index, row in enumerate(context.rows):
                    artwork_analysis = match_artwork_analysis(
                        ai_state.payload["final_output"],
                        row["artiste"],
                        row["tableau"],
                        index,
                    )
                    with analysis_cols[index % len(analysis_cols)]:
                        st.markdown(f"**{row['artiste']}**  \n*{row['tableau']}*")
                        if artwork_analysis:
                            st.markdown(
                                (
                                    "<div style='margin-top:0.2rem; margin-bottom:0.9rem; "
                                    "padding:0.9rem 1rem; border:1px solid rgba(17,17,17,0.08); "
                                    "border-radius:10px; background:rgba(255,255,255,0.55); "
                                    f"color:#1d1d1d; line-height:1.55; font-size:0.92rem;'>{analysis_text_to_html(artwork_analysis)}</div>"
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("Analyse IA non trouvée pour cette œuvre.")

                st.markdown("---")
                st.markdown("**Comparaison stylistique globale**")
                global_analysis = extract_global_analysis(ai_state.payload["final_output"])
                if global_analysis:
                    st.markdown(
                        (
                            "<div style='margin-top:0.2rem; padding:1rem 1.1rem; "
                            "border-left:6px solid #b1221c; border-radius:10px; "
                            "background:rgba(255,255,255,0.6); color:#1d1d1d; "
                            f"line-height:1.6;'>{analysis_text_to_html(global_analysis)}</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("La comparaison stylistique globale n'a pas été trouvée dans la réponse IA.")
            elif ai_state.error is not None:
                st.warning(f"Analyse IA indisponible : {ai_state.error}")

    def _render_summary(self, df_results: pd.DataFrame) -> None:
        st.subheader("Résumé des résultats")
        st.dataframe(df_results, width="stretch", hide_index=True)

    def _render_umap(self, context: ArtworkResultsContext) -> None:
        if self.latent_error is not None:
            st.warning(f"Visualisation UMAP indisponible : {self.latent_error}")
            return
        if self.latent_bundle is None:
            return

        latent_2d, labels, classnames, filenames = self.latent_bundle
        with st.expander("Espace latent (UMAP interactif)", expanded=False):
            # On enrichit les tooltips UMAP avec les similarités de la requête
            # courante afin de relier la recherche locale à l'espace global.
            df_umap = UmapPlotBuilder.build_dataframe(
                latent_2d,
                labels,
                classnames,
                filenames,
                context.results,
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
            UmapPlotBuilder.add_highlights(fig, context, latent_2d, filenames)
            fig.update_layout(
                xaxis_title="Dimension UMAP 1",
                yaxis_title="Dimension UMAP 2",
                legend_title="Styles",
                height=700,
                hoverlabel=dict(font=dict(size=16)),
            )
            st.plotly_chart(fig, width="stretch")

    def _render_gradcam(self, context: ArtworkResultsContext, gradcam_pair_count: int) -> None:
        # Grad-CAM++ reste derrière une interaction explicite car son calcul est
        # coûteux et surtout utile pour les usages d'explicabilité avancée.
        show_gradcam_history = st.checkbox("Grad-CAM history", key="show_gradcam_history")
        available_layers = (
            self.retriever.available_explanation_layers()
            if show_gradcam_history and self.retriever is not None
            else []
        )
        if show_gradcam_history and available_layers:
            self._render_gradcam_history(context, available_layers, gradcam_pair_count)
        else:
            st.info(
                "Active l'option Grad-CAM++ pour visualiser les zones qui contribuent "
                "à la similarité du top-1."
            )

    def _render_gradcam_history(
        self,
        context: ArtworkResultsContext,
        available_layers: list[str],
        gradcam_pair_count: int,
    ) -> None:
        # On échantillonne plusieurs profondeurs du réseau afin de montrer
        # une histoire visuelle de l'explication, pas une seule coupe arbitraire.
        history_layer_numbers = build_random_gradcam_layer_numbers(gradcam_pair_count, 1, 245)
        history_layers, missing_history_layers = select_explanation_layers(
            available_layers,
            history_layer_numbers,
        )
        _, layer_labels = format_explanation_layer_options(available_layers)
        if not history_layers:
            st.warning("Aucune des couches de l'historique n'est disponible pour ce modèle.")
            return

        with st.spinner("Calcul des cartes Grad-CAM++..."):
            history_explanations = [
                (
                    layer_number,
                    self.retriever.explain_similarity(
                        context.query_path,
                        context.best["filepath"],
                        target_layer_name=layer_name,
                    ),
                )
                for layer_number, layer_name in history_layers
            ]

        best_artist, best_title = extract_artist_and_title(context.best["filepath"])
        with st.expander("Explication visuelle (Grad-CAM++)", expanded=False):
            st.markdown(
                (
                    "<div class='museum-card'><strong>Top-1 sélectionné</strong>"
                    f"<em>{best_artist} • {best_title}</em></div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown("**Grad-CAM history**")
            st.caption(
                f"{len(history_explanations)} paires de cartes affichees pour les couches "
                f"{', '.join(str(layer_number) for layer_number in history_layer_numbers)}."
            )
            if missing_history_layers:
                st.caption(
                    "Couches d'historique non disponibles pour ce modèle : "
                    f"{', '.join(str(layer_number) for layer_number in missing_history_layers)}."
                )

            for layer_number, explanation in history_explanations:
                layer_name = explanation["target_layer"]
                layer_label = layer_labels.get(layer_name, layer_name)
                col_query, col_candidate = st.columns(2)
                with col_query:
                    st.image(
                        explanation["query_overlay"],
                        caption=(
                            f"Upload ({explanation['method']}) • couche {layer_number} • "
                            f"{layer_label} • `{layer_name}` • similarité {explanation['similarity']:.3f}"
                        ),
                        width="stretch",
                    )
                with col_candidate:
                    st.image(
                        explanation["candidate_overlay"],
                        caption=(
                            f"Top-1 ({explanation['method']}) • couche {layer_number} • "
                            f"{layer_label} • `{layer_name}` • similarité {explanation['similarity']:.3f}"
                        ),
                        width="stretch",
                    )

    def _render_unavailable_state(self, uploaded) -> None:
        upload_disabled = bool(
            (self.retriever_error is not None)
            or (not bool(self.runtime_status and self.runtime_status["upload_enabled"]))
        )
        if uploaded is None:
            st.info(
                "Upload désactivé tant que le modèle, les embeddings et `data/out` ne sont pas "
                "synchronisés."
                if upload_disabled
                else "Charge une image pour lancer la recherche."
            )
        else:
            st.error("Le moteur n'est pas prêt. Vérifie le modèle, les embeddings et data/out.")


def run_app() -> None:
    setup_page(APP_LOGO_PATH)
    InternalArtworkStore.initialize_session_state()
    ArtXplainApp().run()
