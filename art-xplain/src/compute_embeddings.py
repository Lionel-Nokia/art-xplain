"""Compute gallery embeddings and save a coherent metadata export.

Ce script :
- charge le même encoder Keras que retrieval.py ;
- parcourt un split Keras-ready (train/val/test) sous keras_root ;
- calcule un embedding par image avec la même préparation ;
- sauvegarde vectors, labels, filenames et classnames ensemble.

Structure attendue :
keras_root/
    train/
        style_1/
            img1.jpg
        style_2/
            img2.jpg
    val/
        ...
    test/
        ...

Fichiers produits dans embeddings_root :
- vectors.npy
- labels.npy
- filenames.npy
- classnames.npy
- embeddings_bundle.npz
- embeddings_manifest.json
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import tensorflow as tf

from .utils import load_config, ensure_dir


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _sha1_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def _list_class_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _load_image_for_encoder(image_path: str | Path, img_size: int) -> tf.Tensor:
    """Même logique que dans retrieval.py."""
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    arr = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(arr, 0)


def _compute_embedding(
    encoder: tf.keras.Model,
    image_path: str | Path,
    img_size: int,
) -> np.ndarray:
    batch = _load_image_for_encoder(image_path, img_size)
    emb = encoder(batch, training=False).numpy()

    if emb.ndim != 2 or emb.shape[0] != 1:
        raise ValueError(
            f"Sortie encoder invalide pour {image_path}: shape={emb.shape}, attendu=(1, D)"
        )

    return emb[0].astype(np.float32)


def _resolve_dataset_root(cfg: dict) -> tuple[Path, str]:
    """
    Détermine le dossier source des images pour les embeddings.

    Priorité :
    1. paths.dataset_root si présent
    2. paths.keras_root/test
    3. paths.keras_root/val
    4. paths.keras_root/train
    5. paths.keras_root
    """
    paths_cfg = cfg.get("paths", {})

    if "dataset_root" in paths_cfg:
        root = Path(paths_cfg["dataset_root"])
        return root, "dataset_root"

    if "keras_root" not in paths_cfg:
        raise KeyError(
            "Ni 'paths.dataset_root' ni 'paths.keras_root' n'existent dans config.yaml"
        )

    keras_root = Path(paths_cfg["keras_root"])

    for split in ("test", "val", "train"):
        split_root = keras_root / split
        if split_root.exists() and split_root.is_dir():
            return split_root, f"keras_root/{split}"

    return keras_root, "keras_root"


def main():
    cfg = load_config()

    dataset_root, dataset_source = _resolve_dataset_root(cfg)
    embeddings_root = Path(cfg["paths"]["embeddings_root"])
    models_root = Path(cfg["paths"]["models_root"])
    ensure_dir(embeddings_root)

    img_size = int(cfg["model"]["img_size"])
    encoder_path = models_root / "encoder.keras"

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dossier source introuvable: {dataset_root} (résolu depuis {dataset_source})"
        )

    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder.keras introuvable: {encoder_path}")

    print(f"Source images   : {dataset_root.resolve()} [{dataset_source}]")
    print(f"Embeddings root : {embeddings_root.resolve()}")
    print(f"Encoder path    : {encoder_path.resolve()}")
    print(f"Image size      : {img_size}")

    encoder = tf.keras.models.load_model(
        encoder_path,
        compile=False,
        safe_mode=False,
    )

    class_dirs = _list_class_dirs(dataset_root)
    if not class_dirs:
        raise ValueError(f"Aucun dossier de classe trouvé dans {dataset_root}")

    vectors_list: list[np.ndarray] = []
    labels_list: list[int] = []
    filenames_list: list[str] = []
    classnames_list: list[str] = []
    skipped_files: list[dict[str, str]] = []

    print(f"Nombre de styles détectés : {len(class_dirs)}")

    for class_dir in class_dirs:
        image_files = _list_images(class_dir)
        if not image_files:
            print(f"[WARN] Aucun fichier image dans {class_dir}")
            continue

        class_name = class_dir.name
        class_idx = len(classnames_list)
        classnames_list.append(class_name)

        print(f"-> {class_name}: {len(image_files)} image(s)")

        valid_count = 0
        for img_path in image_files:
            try:
                vec = _compute_embedding(encoder, img_path, img_size)
                vectors_list.append(vec)
                labels_list.append(class_idx)
                filenames_list.append(str(img_path.resolve()))
                valid_count += 1
            except Exception as exc:
                skipped_files.append({
                    "file": str(img_path),
                    "error": str(exc),
                })
                print(f"[WARN] Fichier ignoré: {img_path} -> {exc}")

        print(f"   {valid_count} embedding(s) calculé(s)")

    if not vectors_list:
        raise ValueError("Aucun embedding n'a pu être calculé.")

    vectors = np.stack(vectors_list).astype(np.float32)
    labels = np.asarray(labels_list, dtype=np.int64)
    filenames = np.asarray(filenames_list, dtype=object)
    classnames = np.asarray(classnames_list, dtype=object)

    n_samples = vectors.shape[0]

    if not (n_samples == len(labels) == len(filenames)):
        raise RuntimeError(
            "Incohérence de cardinalité après calcul des embeddings.\n"
            f" - vectors   : {n_samples}\n"
            f" - labels    : {len(labels)}\n"
            f" - filenames : {len(filenames)}"
        )

    if labels.size > 0:
        min_label = int(labels.min())
        max_label = int(labels.max())
        if min_label < 0 or max_label >= len(classnames):
            raise RuntimeError(
                "Incohérence entre labels et classnames.\n"
                f" - min_label   : {min_label}\n"
                f" - max_label   : {max_label}\n"
                f" - num_classes : {len(classnames)}"
            )

    bundle_path = embeddings_root / "embeddings_bundle.npz"
    np.savez_compressed(
        bundle_path,
        vectors=vectors,
        labels=labels,
        filenames=filenames,
        classnames=classnames,
    )

    np.save(embeddings_root / "vectors.npy", vectors)
    np.save(embeddings_root / "labels.npy", labels)
    np.save(embeddings_root / "filenames.npy", filenames)
    np.save(embeddings_root / "classnames.npy", classnames)

    manifest = {
        "num_samples": int(n_samples),
        "embedding_dim": int(vectors.shape[1]),
        "num_classes": int(len(classnames)),
        "img_size": int(img_size),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_source": dataset_source,
        "embeddings_root": str(embeddings_root.resolve()),
        "encoder_path": str(encoder_path.resolve()),
        "vectors_sha1": _sha1_of_array(vectors),
        "labels_sha1": _sha1_of_array(labels),
        "filenames_sha1": _sha1_of_array(
            np.asarray([str(x) for x in filenames], dtype="<U2048")
        ),
        "classnames_sha1": _sha1_of_array(
            np.asarray([str(x) for x in classnames], dtype="<U512")
        ),
        "skipped_count": len(skipped_files),
        "skipped_files": skipped_files[:200],
    }

    manifest_path = embeddings_root / "embeddings_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEmbeddings export terminé")
    print(f" - samples        : {n_samples}")
    print(f" - dim            : {vectors.shape[1]}")
    print(f" - classes        : {len(classnames)}")
    print(f" - bundle         : {bundle_path.resolve()}")
    print(f" - vectors.npy    : {(embeddings_root / 'vectors.npy').resolve()}")
    print(f" - labels.npy     : {(embeddings_root / 'labels.npy').resolve()}")
    print(f" - filenames.npy  : {(embeddings_root / 'filenames.npy').resolve()}")
    print(f" - classnames.npy : {(embeddings_root / 'classnames.npy').resolve()}")
    print(f" - manifest       : {manifest_path.resolve()}")
    print(f" - skipped files  : {len(skipped_files)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
