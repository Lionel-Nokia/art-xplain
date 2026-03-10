"""Create a Keras-style directory dataset from a CSV catalogue.

This script searches a Kaggle-style dataset directory for a CSV file,
matches CSV rows to image files, filters and keeps the top styles, and
materializes train/val/test splits as folders suitable for
`tf.keras.utils.image_dataset_from_directory`.

Key behaviour:
- Auto-detects the images root directory (using a hint or by counting images)
- Picks the first CSV (prioritising filenames containing "class")
- Keeps only styles with sufficient examples and the top-N styles
- Performs stratified train/val/test splits and copies image files into
    `keras_root/{train,val,test}/{style}/...`
"""

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import shutil
import re
import ast
from sklearn.model_selection import train_test_split
from .utils import load_config, ensure_dir

# Image file extensions considered when searching directories
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
# Fixed RNG seed to make splits reproducible
RANDOM_SEED = 42

def find_first_csv(root: Path) -> Path:
    """Find the most appropriate CSV file under `root`.

    Preference is given to filenames that contain the substring "class",
    and then shorter names. Raises `FileNotFoundError` if none are found.
    """

    candidates = list(root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {root}")

    # Sort so files with "class" in the name come first, then by length.
    candidates.sort(key=lambda p: ("class" not in p.name.lower(), len(p.name)))
    return candidates[0]

def sanitize_filename(name: str) -> str:
    """Normalize whitespace in a filename or title.

    Removes leading/trailing whitespace and collapses internal whitespace to
    single spaces. This can help match filenames whose stored names differ by
    accidental spacing.
    """

    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name

def resolve_image_path(images_root: Path, rel: str) -> Path | None:
    p = Path(rel)
    # Attempt multiple heuristics to resolve a CSV-provided path to an actual
    # file on disk. Return `None` if no candidate exists.

    # 1) CSV may already contain a relative path from images_root
    cand = images_root / p
    if cand.exists():
        return cand

    # 2) CSV may store only the filename; try matching by name in images_root
    cand = images_root / p.name
    if cand.exists():
        return cand

    # 3) Try a sanitized filename (collapse stray whitespace)
    s = sanitize_filename(p.name)
    cand = images_root / s
    if cand.exists():
        return cand

    # No candidate found
    return None

def auto_detect_images_root(kaggle_root: Path, images_subdir_hint: str) -> Path:
    """Try to locate the images directory inside a kaggle-style root.

    If `kaggle_root/images_subdir_hint` exists, return it. Otherwise search
    recursively for the directory that contains the largest number of image
    files (by `IMG_EXT`) and return that directory. Raises `FileNotFoundError`
    if no suitable directory is found.
    """

    hinted = kaggle_root / images_subdir_hint
    if hinted.exists():
        return hinted

    best = None
    best_count = 0
    # Walk through directories and count image files to pick the best candidate
    for d in kaggle_root.rglob("*"):
        if d.is_dir():
            count = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            if count > best_count:
                best, best_count = d, count

    if best is None or best_count == 0:
        raise FileNotFoundError("Impossible de trouver un dossier d'images dans kaggle_root")
    return best


def detect_images_root_from_filenames(kaggle_root: Path, images_subdir_hint: str, filenames: pd.Series) -> Path:
    """Choose the image root that resolves the most CSV filenames.

    This is robust to multiple dataset layouts, including:
    - `kaggle_root/images/<style>/<file>.jpg`
    - `kaggle_root/<style>/<file>.jpg`
    """

    candidates = []
    hinted = kaggle_root / images_subdir_hint
    if images_subdir_hint and hinted.exists():
        candidates.append(hinted)

    if kaggle_root.exists():
        candidates.append(kaggle_root)
        candidates.extend(sorted(p for p in kaggle_root.iterdir() if p.is_dir()))

    # Keep insertion order while removing duplicates
    seen = set()
    uniq_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq_candidates.append(c)

    sample = filenames.astype(str).head(500)
    best = None
    best_hits = -1
    for cand in uniq_candidates:
        hits = sum(resolve_image_path(cand, rel) is not None for rel in sample)
        if hits > best_hits:
            best = cand
            best_hits = hits

    if best is None or best_hits <= 0:
        # Fallback for legacy behavior if filename probing fails
        return auto_detect_images_root(kaggle_root, images_subdir_hint)
    return best


def infer_label_from_filename_parent(filename_series: pd.Series) -> pd.Series | None:
    """Infer labels from parent folder in filename paths when possible."""

    parents = filename_series.astype(str).map(lambda x: Path(x).parent.as_posix())
    valid = parents[parents != "."]
    if len(valid) < 0.9 * len(parents):
        return None
    return valid


def normalize_label_value(value: str) -> str:
    """Normalize label values from CSV (including list-like strings)."""

    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            # ast.literal_eval() attempt to parse list-like strings
            # (e.g. "['Abstract Expressionism']") and take the first element
            # ast.literal_eval("['Abstract Expressionism']")  # -> ['Abstract Expressionism']
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list) and parsed:
                text = str(parsed[0]).strip()
        except (SyntaxError, ValueError):
            pass
    return text.replace("/", "_")

def materialize_split(split_df: pd.DataFrame, split: str, out_root: Path, images_root: Path,
                      filename_col: str, label_col: str) -> tuple[int, int]:
    """Copy dataset rows into `out_root/{split}/{label}/` folders.

    For each row in `split_df`, attempt to resolve the image file and copy it
    into a directory named after the label (with '/' replaced by '_'). The
    function returns a tuple `(copied, missing)` indicating how many files were
    copied and how many rows could not be resolved to an image file.
    """

    missing = 0
    copied = 0
    for _, row in split_df.iterrows():
        # Create a safe directory name for the style/label
        style = str(row[label_col]).replace("/", "_")
        rel = str(row[filename_col])

        # Resolve the path on disk using heuristics
        src = resolve_image_path(images_root, rel)
        if src is None:
            missing += 1
            continue

        dst_dir = out_root / split / style
        ensure_dir(dst_dir)
        dst = dst_dir / src.name

        # Copy only when target does not already exist (idempotent behavior)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    return copied, missing

def clean_output_root(out_root: Path) -> None:
    """Delete all generated files/folders under `out_root`."""

    if not out_root.exists():
        print(f"Clean: rien à supprimer, dossier absent: {out_root}")
        return

    removed = 0
    for child in out_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            removed += 1
        else:
            child.unlink()
            removed += 1
    print(f"Clean: {removed} élément(s) supprimé(s) dans {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Keras train/val/test splits from a Kaggle-style CSV dataset."
    )
    parser.add_argument(
        "--clean-out",
        action="store_true",
        help="Supprime tout le contenu de paths.keras_root avant de régénérer les splits.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Supprime tout le contenu de paths.keras_root puis quitte sans générer les splits.",
    )
    return parser.parse_args()
