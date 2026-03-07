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
import pandas as pd
import shutil
import re
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

def main():
    """Entry point: read config, filter CSV, split and materialize files.

    This function coordinates: locating the CSV, selecting the label column,
    filtering styles by frequency, creating stratified splits, and writing
    images into the output `keras_root` folder in a structure compatible with
    `tf.keras` image utilities.
    """

    cfg = load_config()
    kaggle_root = Path(cfg["paths"]["kaggle_root"])
    out_root = Path(cfg["paths"]["keras_root"])
    images_hint = cfg["paths"].get("images_subdir_hint", "images")

    # Dataset configuration parameters
    keep_top_styles = int(cfg["dataset"]["keep_top_styles"])
    min_images_per_style = int(cfg["dataset"]["min_images_per_style"])
    test_size = float(cfg["dataset"]["test_size"])
    val_size = float(cfg["dataset"]["val_size"])

    # Load the CSV catalogue and detect columns (case-insensitive matching)
    csv_path = find_first_csv(kaggle_root)
    df = pd.read_csv(csv_path)

    cols = {c.lower(): c for c in df.columns}
    if "filename" not in cols:
        raise ValueError(f"CSV {csv_path.name}: colonne 'filename' introuvable. Colonnes={list(df.columns)}")

    # Choose a label column from common candidates
    label_col = None
    for cand in ["style", "movement", "genre", "artist"]:
        if cand in cols:
            label_col = cols[cand]
            break
    if label_col is None:
        raise ValueError("Aucune colonne label trouvée parmi: style/movement/genre/artist")

    filename_col = cols["filename"]

    # Locate images directory using hint or auto-detection
    images_root = auto_detect_images_root(kaggle_root, images_hint)

    # Keep only the filename and label columns, drop rows with missing values
    df = df[[filename_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(str)

    # Filter styles by minimum count and keep only the top-N frequent styles
    counts = df[label_col].value_counts()
    keep = counts[counts >= min_images_per_style].head(keep_top_styles).index
    df = df[df[label_col].isin(keep)].copy()

    print("CSV:", csv_path)
    print("Images root:", images_root)
    print("Label:", label_col)
    print("Styles gardés:", list(keep))
    print("Total rows after filtering:", len(df))

    # Create stratified train/val/test splits preserving class proportions
    train_df, tmp_df = train_test_split(
        df, test_size=(test_size + val_size),
        random_state=RANDOM_SEED, stratify=df[label_col]
    )
    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        tmp_df, test_size=(1 - rel_val),
        random_state=RANDOM_SEED, stratify=tmp_df[label_col]
    )

    # Ensure target folders exist
    ensure_dir(out_root / "train")
    ensure_dir(out_root / "val")
    ensure_dir(out_root / "test")

    # Materialize each split by copying resolved image files into label folders
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        copied, missing = materialize_split(
            split_df, split_name, out_root, images_root, filename_col, label_col
        )
        print(f"{split_name}: copied={copied} missing={missing} total_rows={len(split_df)}")

if __name__ == "__main__":
    main()
