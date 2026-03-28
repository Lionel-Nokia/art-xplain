from __future__ import annotations

import json
import re
import unicodedata
from html import escape
from pathlib import Path


def normalize_lookup_text(value: object) -> str:
    # La réponse IA n'est pas toujours strictement stable :
    # accents, tirets, ponctuation ou variantes typographiques
    # peuvent empêcher un matching exact entre une œuvre du DataFrame
    # et un chapitre renvoyé par l'agent.
    # Cette normalisation fournit donc une clé de comparaison plus robuste.
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    for char in ["-", "–", "—", "_", "/", ",", ":", ";", ".", "(", ")", "'", '"']:
        text = text.replace(char, " ")
    return " ".join(text.split())


def match_normalized_text(expected: str, candidate: str) -> bool:
    if not expected or not candidate:
        return False
    if expected == candidate:
        return True
    if expected in candidate or candidate in expected:
        return True

    expected_tokens = expected.split()
    candidate_tokens = candidate.split()
    return bool(expected_tokens and all(token in candidate_tokens for token in expected_tokens))


def token_overlap_score(reference: str, candidate: str) -> float:
    reference_tokens = set(reference.split())
    candidate_tokens = set(candidate.split())
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = reference_tokens.intersection(candidate_tokens)
    return len(overlap) / max(len(reference_tokens), 1) if overlap else 0.0


def strip_http_links(text: str) -> str:
    # On retire les URLs pour garder un rendu Streamlit lisible
    # et éviter de polluer les blocs analytiques avec des liens bruts
    # parfois ajoutés par les modèles génératifs.
    cleaned = str(text)
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"\((https?://[^)]+)\)", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\(\s+\)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_analysis_text(value: object) -> str:
    # L'agent peut renvoyer du texte libre, des listes ou des dictionnaires JSON.
    # Ce helper a pour responsabilité unique de convertir ces formes hétérogènes
    # en texte affichable de manière déterministe dans l'UI.
    if value is None:
        return ""
    if isinstance(value, str):
        return strip_http_links(value.strip())
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for part in (format_analysis_text(item) for item in value) if part)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = format_analysis_text(item)
            if item_text:
                parts.append(f"{key} : {item_text}")
        return strip_http_links("\n".join(parts))
    return strip_http_links(str(value).strip())


def analysis_text_to_html(text: str) -> str:
    # On reste volontairement sur une transformation HTML légère :
    # - échappement du contenu pour éviter l'injection,
    # - prise en charge de quelques marqueurs markdown simples,
    # - conservation des retours à la ligne pour la lisibilité.
    html = escape(str(text).strip())
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    for subtitle in ["Contexte historique et technique", "Spécificités stylistiques"]:
        html = html.replace(subtitle, f"<strong>{subtitle}</strong>")
    return html.replace("\n", "<br>")


def split_analysis_title(title: object) -> tuple[str, str]:
    title_text = str(title).strip()
    for separator in [" - ", " – ", " — "]:
        if separator in title_text:
            artist, artwork = title_text.split(separator, 1)
            return artist.strip(), artwork.strip()
    return "", title_text


def is_global_analysis_title(title: object) -> bool:
    normalized_title = normalize_lookup_text(title)
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


def clean_analysis_subtitle(text: object) -> str:
    subtitle = str(text).strip()
    if normalize_lookup_text(subtitle) in {"contexte historique et stylistique", "contexte historique"}:
        return ""
    return subtitle


def format_chapter_content(chapter_content: object) -> str:
    if not isinstance(chapter_content, list):
        return format_analysis_text(chapter_content)

    blocks: list[str] = []
    for item in chapter_content:
        if isinstance(item, dict):
            subtitle = clean_analysis_subtitle(item.get("sous_titre", ""))
            text = str(item.get("texte", "")).strip()
            if subtitle and text:
                blocks.append(f"{subtitle}\n{text}")
            elif subtitle:
                blocks.append(subtitle)
            elif text:
                blocks.append(text)
        else:
            text = format_analysis_text(item)
            if text:
                blocks.append(text)
    return "\n\n".join(block for block in blocks if block)


def extract_json_payload(final_output: str) -> dict[str, object] | list[object] | None:
    # Le contrat de sortie IA n'est pas parfaitement fiable :
    # selon le prompt ou le modèle, le JSON peut arriver brut
    # ou encapsulé dans un bloc markdown ```json ... ```.
    # On essaie donc plusieurs candidats avant de conclure à un échec de parsing.
    raw_text = str(final_output).strip()
    candidates = [raw_text]
    if "```" in raw_text:
        for block in raw_text.split("```"):
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


def coerce_payload_to_chapters(
    payload: dict[str, object] | list[object] | None,
) -> list[dict[str, object]] | None:
    # Cette fonction sert d'adaptateur entre plusieurs formats de réponse IA
    # et la structure "chapitres" attendue par le front.
    # En revue, c'est une zone clé : elle absorbe la variabilité du modèle
    # pour éviter de disperser des `if format_x / if format_y` partout dans l'app.
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
                chapter_title = " - ".join(part for part in [artist, title] if part).strip() or title or artist or "Oeuvre"

                raw_analysis = work.get("analyse")
                if isinstance(raw_analysis, dict):
                    content = [
                        {
                            "sous_titre": str(key).replace("_", " ").strip().capitalize(),
                            "texte": format_analysis_text(value),
                        }
                        for key, value in raw_analysis.items()
                        if format_analysis_text(value)
                    ]
                else:
                    content = []
                    for key in ["contexte_historique", "specificites_stylistiques", "elements_techniques_marquants"]:
                        value = format_analysis_text(work.get(key))
                        if value:
                            content.append(
                                {
                                    "sous_titre": key.replace("_", " ").strip().capitalize(),
                                    "texte": value,
                                }
                            )
                normalized_chapters.append({"titre": chapter_title, "contenu": content or format_analysis_text(work)})

            for global_candidate in [
                payload.get("comparaison_stylistique_globale"),
                payload.get("rapprochement_stylistique"),
                payload.get("analyse_stylistique_globale"),
            ]:
                global_text = format_analysis_text(global_candidate)
                if global_text:
                    normalized_chapters.append(
                        {"titre": "Analyse stylistique comparative", "contenu": global_text}
                    )
                    break
            return normalized_chapters or None

    if isinstance(payload, list):
        return [chapter for chapter in payload if isinstance(chapter, dict)] or None
    return None


def extract_chapters_payload(final_output: str) -> list[dict[str, object]] | None:
    return coerce_payload_to_chapters(extract_json_payload(final_output))


def match_artwork_analysis(
    final_output: str,
    artist: str,
    title: str,
    result_index: int | None = None,
) -> str | None:
    # Le matching est volontairement tolérant :
    # on tente d'abord un rapprochement fort artiste/titre,
    # puis on retombe sur un score par recouvrement de tokens.
    # Cela couvre les cas où l'IA reformule légèrement les noms d'œuvres.
    del result_index
    chapters = extract_chapters_payload(final_output)
    title_key = normalize_lookup_text(title)
    artist_key = normalize_lookup_text(artist)
    if chapters is None:
        return None

    best_score = 0.0
    best_content: str | None = None
    for chapter in chapters:
        if not isinstance(chapter, dict) or is_global_analysis_title(chapter.get("titre", "")):
            continue

        chapter_artist, chapter_title = split_analysis_title(chapter.get("titre", ""))
        normalized_chapter_artist = normalize_lookup_text(chapter_artist)
        normalized_chapter_title = normalize_lookup_text(chapter_title)

        if match_normalized_text(artist_key, normalized_chapter_artist) and match_normalized_text(
            title_key, normalized_chapter_title
        ):
            return format_chapter_content(chapter.get("contenu", []))

        score = 0.0
        if title_key:
            score += max(
                token_overlap_score(title_key, normalized_chapter_title),
                token_overlap_score(title_key, normalize_lookup_text(chapter.get("titre", ""))),
            )
        if artist_key:
            artist_score = max(
                token_overlap_score(artist_key, normalized_chapter_artist),
                token_overlap_score(artist_key, normalize_lookup_text(chapter.get("titre", ""))),
            )
            score += min(artist_score, 1.0)
        if chapter_artist and chapter_title:
            score += 0.1
        if score > best_score:
            best_score = score
            best_content = format_chapter_content(chapter.get("contenu", []))

    # Le seuil évite d'afficher une "mauvaise" analyse juste parce qu'un titre
    # partage quelques mots. En cas d'ambiguïté, on préfère afficher rien
    # plutôt qu'induire l'utilisateur en erreur.
    return best_content if best_score >= 0.8 else None


def extract_global_analysis(final_output: str) -> str | None:
    chapters = extract_chapters_payload(final_output)
    if chapters is None:
        return None
    for chapter in chapters:
        if isinstance(chapter, dict) and is_global_analysis_title(chapter.get("titre", "")):
            return format_chapter_content(chapter.get("contenu", []))
    return None


def match_source_artwork_analysis(
    final_output: str,
    source_artist: str,
    source_title: str,
    artwork_of_interest: str,
    source_display_name: str | None = None,
) -> str | None:
    # L'œuvre source est un cas à part :
    # elle peut être nommée à partir du nom du fichier uploadé,
    # d'un titre reconstruit, ou du champ `artwork_of_interest`.
    # On croise donc plusieurs clés candidates pour maximiser les chances
    # de retrouver le bon chapitre dans la réponse IA.
    direct_match = match_artwork_analysis(final_output, source_artist, source_title)
    if direct_match:
        return direct_match

    chapters = extract_chapters_payload(final_output)
    if chapters is None:
        return None

    source_title_key = normalize_lookup_text(source_title)
    interest_key = normalize_lookup_text(artwork_of_interest)
    display_key = normalize_lookup_text(Path(str(source_display_name or "")).stem)
    source_artist_key = normalize_lookup_text(source_artist)
    candidate_keys = [key for key in [source_title_key, interest_key, display_key] if key]

    best_score = 0.0
    best_content: str | None = None
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_title = str(chapter.get("titre", "")).strip()
        normalized_chapter_title = normalize_lookup_text(chapter_title)
        if is_global_analysis_title(chapter_title):
            continue

        chapter_artist, chapter_artwork = split_analysis_title(chapter_title)
        normalized_chapter_artist = normalize_lookup_text(chapter_artist)
        normalized_chapter_artwork = normalize_lookup_text(chapter_artwork)

        if any(
            match_normalized_text(candidate_key, normalized_chapter_title)
            or match_normalized_text(candidate_key, normalized_chapter_artwork)
            for candidate_key in candidate_keys
        ):
            return format_chapter_content(chapter.get("contenu", []))

        score = 0.0
        for candidate_key in candidate_keys:
            score = max(
                score,
                token_overlap_score(candidate_key, normalized_chapter_title),
                token_overlap_score(candidate_key, normalized_chapter_artwork),
            )

        if source_artist_key and normalized_chapter_artist:
            if match_normalized_text(source_artist_key, normalized_chapter_artist):
                score += 0.2
            else:
                score += min(token_overlap_score(source_artist_key, normalized_chapter_artist), 0.2)

        if score > best_score:
            best_score = score
            best_content = format_chapter_content(chapter.get("contenu", []))

    return best_content if best_score >= 0.55 else None
