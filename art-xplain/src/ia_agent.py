from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display, Markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_file(dotenv_path: str | Path | None = None) -> None:
    """
    Charge un fichier .env simple dans les variables d'environnement.
    """
    path = Path(dotenv_path) if dotenv_path is not None else PROJECT_ROOT.parent / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str = "config_agent.yaml") -> dict:
    """
    Charge le fichier YAML.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_ai_config(config: dict) -> dict:
    """
    Récupère la section 'ai-agent' du YAML.
    """
    if "ai-agent" not in config:
        raise ValueError("Section 'ai-agent' manquante dans config.yaml")
    return config["ai-agent"]


def sanitize_filename(value: str) -> str:
    """
    Nettoie une chaîne pour en faire un nom de fichier correct.
    """
    value = value.strip().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    cleaned = "".join(c for c in value if c in allowed or c == "_")
    return cleaned[:120] if cleaned else "analysis"


def dataframe_to_works_list(df: pd.DataFrame) -> str:
    """
    Transforme le DataFrame en liste textuelle d'oeuvres.
    Colonnes attendues : Artiste, Titre, Année
    """
    required_columns = {"Artiste", "Titre", "Année"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")

    lines = []
    for _, row in df.iterrows():
        artiste = str(row["Artiste"]).strip()
        titre = str(row["Titre"]).strip()
        annee = row["Année"]

        if pd.notna(annee):
            try:
                annee_int = int(float(annee))
                lines.append(f"- {artiste}\t{titre} {annee_int}")
            except (ValueError, TypeError):
                lines.append(f"- {artiste}\t{titre}")
        else:
            lines.append(f"- {artiste}\t{titre}")

    return "\n".join(lines)


def build_message_from_dataframe(
    df: pd.DataFrame,
    artwork_of_interest: str,
    prompt_template: str,
) -> str:
    """
    Construit le message utilisateur à partir du DataFrame.
    """
    works_list = dataframe_to_works_list(df)
    return prompt_template.format(
        works_list=works_list,
        artwork_of_interest=artwork_of_interest,
    )


def create_agent_from_config(
    ai_config: dict,
    profile_name: Optional[str] = None,
) -> Tuple[Agent, dict]:
    """
    Crée l'agent à partir de la configuration YAML.
    """
    try:
        from agents import Agent, WebSearchTool
        from agents.model_settings import ModelSettings
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger le SDK OpenAI Agents. "
            "Le package `agents` actuellement installé ne semble pas être le bon "
            "ou bien ses dépendances sont incompatibles. "
            "Installe `openai-agents` et supprime le package `agents` incompatible si nécessaire."
        ) from exc

    agent_cfg = ai_config.get("agent", {})
    profiles = ai_config.get("profiles", {})

    if profile_name is None:
        profile_name = ai_config.get("default_profile")

    if not profile_name:
        raise ValueError("Aucun profil fourni et 'default_profile' absent du YAML.")

    if profile_name not in profiles:
        raise ValueError(
            f"Profil inconnu : {profile_name}. "
            f"Profils disponibles : {list(profiles.keys())}"
        )

    profile = profiles[profile_name]

    agent = Agent(
        name=agent_cfg.get("name", "Search agent"),
        instructions=profile["instructions"],
        tools=[
            WebSearchTool(
                search_context_size=agent_cfg.get("search_context_size", "low")
            )
        ],
        model=agent_cfg.get("model", "gpt-4o-mini"),
        model_settings=ModelSettings(
            tool_choice=agent_cfg.get("tool_choice", "required")
        ),
    )

    return agent, profile


def save_output(
    content: str,
    prefix: str = "analysis",
    folder: str = "outputs",
    pretty_json: bool = True,
) -> str:
    """
    Sauvegarde le contenu dans un fichier horodaté.
    Si pretty_json=True, tente de reformater le JSON.
    """
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = sanitize_filename(prefix)
    filepath = os.path.join(folder, f"{safe_prefix}_{timestamp}.json")

    if pretty_json:
        try:
            parsed = json.loads(content)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            return filepath
        except json.JSONDecodeError:
            pass

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def list_profiles(config_path: str = "config_agent.yaml") -> list[str]:
    """
    Liste les profils disponibles dans le YAML.
    """
    config = load_config(config_path)
    ai_config = get_ai_config(config)
    return list(ai_config.get("profiles", {}).keys())


async def run_analysis(
    df: pd.DataFrame,
    artwork_of_interest: str,
    config_path: str = "config_agent.yaml",
    profile_name: Optional[str] = None,
    output_folder: str = "outputs",
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Exécute l'analyse :
    - lit la config
    - crée l'agent
    - construit le message
    - lance Runner.run
    - sauvegarde result.final_output
    """
    load_dotenv_file()

    config = load_config(config_path)
    ai_config = get_ai_config(config)

    agent, profile = create_agent_from_config(ai_config, profile_name)
    try:
        from agents import Runner, trace
    except Exception as exc:
        raise RuntimeError(
            "Impossible de lancer l'analyse OpenAI. "
            "Le SDK attendu est `openai-agents`, mais l'environnement charge un autre package `agents`."
        ) from exc

    message = build_message_from_dataframe(
        df=df,
        artwork_of_interest=artwork_of_interest,
        prompt_template=profile["user_prompt_template"],
    )

    with trace("Search"):
        result = await Runner.run(agent, message)

    output_path = ""
    if save_to_file:
        output_path = save_output(
            content=result.final_output,
            prefix=artwork_of_interest,
            folder=output_folder,
            pretty_json=True,
        )

    return {
        "result": result,
        "message": message,
        "output_path": output_path,
        "profile_used": profile_name or ai_config.get("default_profile"),
    }


async def main():
    df = pd.DataFrame([
        {"Artiste": "Tia Peltz", "Titre": "The Tamer", "Année": None},
        {"Artiste": "Amedeo Modigliani", "Titre": "Monsieur Lepoutre", "Année": 1916.0},
        {"Artiste": "Edvard Munch", "Titre": "Self Portrait With Brushes", "Année": 1904.0},
        {"Artiste": "Lucian Freud", "Titre": "David Hockney", "Année": None},
    ])

    artwork_of_interest = "Guy Rose Girl In A Wickford Garden New England"

    analysis = await run_analysis(
        df=df,
        artwork_of_interest=artwork_of_interest,
        config_path="config_agent.yaml",
        profile_name="guide_musée",
        output_folder="outputs",
    )

    print(f"Profil utilisé : {analysis['profile_used']}")
    print(f"Résultat sauvegardé dans : {analysis['output_path']}")
    display(Markdown(analysis["result"].final_output))


# Pour un script Python classique :
# if __name__ == "__main__":
#     asyncio.run(main())
