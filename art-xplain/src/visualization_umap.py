"""Compute a 2D UMAP projection and save a consistent metadata bundle.

This script loads vectors and their associated metadata from embeddings_root,
validates that they all have the same cardinality, computes a 2D UMAP
projection, and saves a coherent export for downstream Streamlit usage.
"""

from __future__ import annotations

from pathlib import Path
import json
import hashlib
import numpy as np
import umap

from .utils import ensure_dir, load_config, relativize_project_path, resolve_project_path, resolve_stored_path


def _sha1_of_array(arr: np.ndarray) -> str:
    """Return a short SHA1 fingerprint for a numpy array."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def _load_required_array(path: Path, *, allow_pickle: bool = False) -> np.ndarray:
    """Load a required numpy file and raise a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def main():
    """
    Build a coherent UMAP export from vectors + metadata.

    Required input files in embeddings_root:
    - vectors.npy
    - labels.npy
    - filenames.npy
    - classnames.npy

    Produced files:
    - latent_2d.npy
    - umap_bundle.npz
    - labels.npy          (rewritten for coherence)
    - filenames.npy       (rewritten for coherence)
    - classnames.npy      (rewritten for coherence)
    - umap_manifest.json
    """
    cfg = load_config()
    emb_root = resolve_project_path(cfg["paths"]["embeddings_root"])
    ensure_dir(emb_root)

    vectors_path = emb_root / "vectors.npy"
    labels_path = emb_root / "labels.npy"
    filenames_path = emb_root / "filenames.npy"
    classnames_path = emb_root / "classnames.npy"

    vectors = _load_required_array(vectors_path)
    labels = _load_required_array(labels_path)
    filenames = _load_required_array(filenames_path, allow_pickle=True)
    classnames = _load_required_array(classnames_path, allow_pickle=True)

    vectors = np.asarray(vectors)
    labels = np.asarray(labels)
    filenames = np.asarray(
        [relativize_project_path(resolve_stored_path(fp)) for fp in filenames],
        dtype=object,
    )
    classnames = np.asarray(classnames, dtype=object)

    if vectors.ndim != 2:
        raise ValueError(
            f"`vectors.npy` doit être de forme (N, D), reçu: {vectors.shape}"
        )

    n_vectors = vectors.shape[0]
    n_labels = len(labels)
    n_filenames = len(filenames)

    if not (n_vectors == n_labels == n_filenames):
        raise ValueError(
            "Incohérence de cardinalité détectée avant calcul UMAP.\n"
            f" - vectors    : {n_vectors}\n"
            f" - labels     : {n_labels}\n"
            f" - filenames  : {n_filenames}\n"
            "Tous ces fichiers doivent provenir du même export amont."
        )

    # Vérification minimale des labels par rapport aux classnames
    if labels.size > 0:
        if np.issubdtype(labels.dtype, np.integer):
            min_label = int(labels.min())
            max_label = int(labels.max())
            if min_label < 0 or max_label >= len(classnames):
                raise ValueError(
                    "labels.npy contient des indices incompatibles avec classnames.npy.\n"
                    f" - min label  : {min_label}\n"
                    f" - max label  : {max_label}\n"
                    f" - classnames : {len(classnames)} classes"
                )

    reducer = umap.UMAP(
        n_neighbors=int(cfg["umap"]["n_neighbors"]),
        min_dist=float(cfg["umap"]["min_dist"]),
        metric="cosine",
        random_state=42,
    )

    latent_2d = reducer.fit_transform(vectors)

    if latent_2d.ndim != 2 or latent_2d.shape != (n_vectors, 2):
        raise RuntimeError(
            f"Sortie UMAP inattendue: {latent_2d.shape}, attendu: ({n_vectors}, 2)"
        )

    # 1) Bundle unique recommandé
    bundle_path = emb_root / "umap_bundle.npz"
    np.savez_compressed(
        bundle_path,
        latent_2d=latent_2d,
        labels=labels,
        filenames=filenames,
        classnames=classnames,
    )

    # 2) Fichiers legacy réécrits dans la même passe
    np.save(emb_root / "latent_2d.npy", latent_2d)
    np.save(emb_root / "labels.npy", labels)
    np.save(emb_root / "filenames.npy", filenames)
    np.save(emb_root / "classnames.npy", classnames)

    # 3) Manifest de contrôle utile pour debug / traçabilité
    manifest = {
        "num_samples": int(n_vectors),
        "vector_dim": int(vectors.shape[1]),
        "num_classes": int(len(classnames)),
        "umap_shape": list(latent_2d.shape),
        "vectors_sha1": _sha1_of_array(vectors),
        "labels_sha1": _sha1_of_array(labels),
        "filenames_sha1": _sha1_of_array(
            np.array([str(x) for x in filenames], dtype="<U512")
        ),
        "classnames_sha1": _sha1_of_array(
            np.array([str(x) for x in classnames], dtype="<U256")
        ),
    }

    manifest_path = emb_root / "umap_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("UMAP bundle saved successfully")
    print(f" - bundle    : {bundle_path.resolve()}")
    print(f" - latent_2d : {(emb_root / 'latent_2d.npy').resolve()} {latent_2d.shape}")
    print(f" - manifest  : {manifest_path.resolve()}")
    print(f" - samples   : {n_vectors}")


if __name__ == "__main__":
    main()
