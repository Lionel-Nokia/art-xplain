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


def print_step(step_number: int, title: str) -> None:
    separator = "=" * 72
    print(f"\n{separator}")
    print(f"build_dataset_from_csv: {step_number} - {title}")
    print(separator)


def print_style_report(df: pd.DataFrame, label_col: str) -> None:
    counts = df[label_col].value_counts()
    if counts.empty:
        print("Aucun style retenu.")
        return

    width = max(len(str(label)) for label in counts.index) + 2
    for label, count in counts.items():
        print(f"{label:<{width}} {count}")


def limit_per_class(df, label_col, max_per_class):
    # Sample each class independently without triggering the pandas groupby.apply warning.
    return (
        df.groupby(label_col, group_keys=False)[df.columns]
          .apply(lambda x: x.sample(min(len(x), max_per_class)))
    )

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
        "--config",
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration YAML.",
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


def prepare_label_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build a normalized `(filename, __label__)` dataframe from the raw CSV."""

    cols = {c.lower(): c for c in df_raw.columns}
    if "filename" not in cols:
        raise ValueError(f"Colonne 'filename' introuvable. Colonnes={list(df_raw.columns)}")

    filename_col = cols["filename"]
    df = df_raw.copy()
    label_col = "__label__"

    derived_labels = infer_label_from_filename_parent(df[filename_col])
    if derived_labels is not None:
        df[label_col] = derived_labels
        print("Label source: parent folder du filename")
    else:
        selected_label_col = None
        for cand in ["style", "movement", "genre", "artist"]:
            if cand in cols:
                selected_label_col = cols[cand]
                break

        if selected_label_col is None:
            raise ValueError("Aucune colonne label trouvée parmi: style/movement/genre/artist")

        df[label_col] = df[selected_label_col]
        print(f"Label source: colonne CSV '{selected_label_col}'")

    df = df[[filename_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(str).map(normalize_label_value)
    return df, filename_col


def split_dataset(
    df_filtered: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test stratified splits using config ratios."""

    train_df, tmp_df = train_test_split(
        df_filtered,
        test_size=(test_size + val_size),
        random_state=RANDOM_SEED,
        stratify=df_filtered[label_col],
    )

    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=(1 - rel_val),
        random_state=RANDOM_SEED,
        stratify=tmp_df[label_col],
    )
    return train_df, val_df, test_df


def build_dataset(config_path: str | Path = "config/config.yaml", *, clean_out: bool = False, clean_only: bool = False) -> None:
    """Run the full dataset build pipeline from config."""

    print_step(1, "Chargement de la configuration")
    cfg = load_config(config_path)
    kaggle_root = Path(cfg["paths"]["kaggle_root"]).expanduser()
    out_root = Path(cfg["paths"]["keras_root"]).expanduser()

    if not kaggle_root.is_absolute():
        kaggle_root = (Path.cwd() / kaggle_root).resolve()
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()

    images_hint = cfg["paths"].get("images_subdir_hint", "images")

    print("kaggle_root =", kaggle_root)
    print("out_root    =", out_root)

    if clean_out or clean_only:
        print_step(2, "Nettoyage du dossier de sortie")
        clean_output_root(out_root)
        if clean_only:
            print("\nExecution terminee apres nettoyage.")
            return

    print_step(3, "Lecture du catalogue CSV")
    csv_path = find_first_csv(kaggle_root)
    df_raw = pd.read_csv(csv_path)
    print("CSV:", csv_path)
    print("Shape:", df_raw.shape)
    print("Columns:", list(df_raw.columns))

    print_step(4, "Preparation des labels")
    df, filename_col = prepare_label_dataframe(df_raw)
    label_col = "__label__"
    print("Colonne image:", filename_col)
    print("Colonne label:", label_col)

    print_step(5, "Filtrage du dataset")
    images_root = detect_images_root_from_filenames(kaggle_root, images_hint, df[filename_col])
    keep_top_styles = int(cfg["dataset"]["keep_top_styles"])
    min_images_per_style = int(cfg["dataset"]["min_images_per_style"])
    max_images = cfg["dataset"].get("max_images")
    test_size = float(cfg["dataset"]["test_size"])
    val_size = float(cfg["dataset"]["val_size"])

    counts = df[label_col].value_counts()
    keep = counts[counts >= min_images_per_style].head(keep_top_styles).index
    df_filtered = df[df[label_col].isin(keep)].copy()
    if max_images is not None:
        df_filtered = limit_per_class(df_filtered, label_col, int(max_images))

    print("Images root:", images_root)
    print("Styles gardes:", list(keep))
    print("Total rows apres filtrage:", len(df_filtered))

    print_step(6, "Creation des splits train/val/test")
    train_df, val_df, test_df = split_dataset(
        df_filtered,
        label_col,
        test_size,
        val_size,
    )
    print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

    print_step(7, "Materialisation des fichiers")
    ensure_dir(out_root / "train")
    ensure_dir(out_root / "val")
    ensure_dir(out_root / "test")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        copied, missing = materialize_split(
            split_df, split_name, out_root, images_root, filename_col, label_col
        )
        print(f"{split_name}: copied={copied} missing={missing} total_rows={len(split_df)}")
    print("- Les missing sont les lignes pour lesquelles le script n'a pas trouve le fichier image source")

    print_step(8, "Resume final")
    print('train:', len(train_df), 'val:', len(val_df), 'test:', len(test_df))
    print("Sortie:", out_root)
    print("\nStyles retenus:")
    print_style_report(df_filtered, label_col)


def main() -> None:
    args = parse_args()
    build_dataset(args.config, clean_out=args.clean_out, clean_only=args.clean_only)


if __name__ == "__main__":
    main()
