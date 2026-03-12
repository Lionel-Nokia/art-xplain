"""Compute and save embeddings with robust split/style label handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .utils import ensure_dir, load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_DIRS = ("train", "val", "test")


def _iter_images(style_dir: Path):
    for p in sorted(style_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p


def collect_records(keras_root: Path):
    """Collect image paths and style names from either split or flat layouts."""

    split_paths = [keras_root / split for split in SPLIT_DIRS if (keras_root / split).is_dir()]
    paths: list[str] = []
    styles: list[str] = []

    if split_paths:
        for split_root in split_paths:
            for style_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
                style_name = style_dir.name
                for img_path in _iter_images(style_dir):
                    paths.append(str(img_path))
                    styles.append(style_name)
    else:
        for style_dir in sorted(p for p in keras_root.iterdir() if p.is_dir()):
            style_name = style_dir.name
            for img_path in _iter_images(style_dir):
                paths.append(str(img_path))
                styles.append(style_name)

    if not paths:
        raise FileNotFoundError(
            f"Aucune image trouvée sous {keras_root}. "
            "Attendu: `keras_root/train|val|test/<style>/*.jpg` ou `keras_root/<style>/*.jpg`."
        )

    class_names = sorted(set(styles))
    style_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([style_to_idx[s] for s in styles], dtype=np.int32)
    file_paths = np.array(paths, dtype=object)
    return file_paths, labels, np.array(class_names, dtype=object)


def build_dataset(file_paths: np.ndarray, labels: np.ndarray, img_size: int, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((file_paths.astype(str), labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
