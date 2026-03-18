
"""Compute a 2D UMAP projection of saved embedding vectors.

This script loads `vectors.npy` from the configured `embeddings_root`, runs
UMAP with parameters taken from the configuration, and writes `latent_2d.npy`.
The produced array can be used for visualization in the Streamlit app.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import umap
from .utils import load_config, ensure_dir


def main():
    """Load vectors, fit UMAP, and save 2D coordinates.

    UMAP parameters are read from the `umap` section of the configuration
    (e.g., `n_neighbors`, `min_dist`). The `metric` is fixed to cosine to
    better match L2-normalized embedding similarity semantics.
    """

    cfg = load_config()
    emb_root = Path(cfg["paths"]["embeddings_root"])
    ensure_dir(emb_root)

    # Load high-dimensional vectors previously computed by `compute_embeddings.py`
    vectors = np.load(emb_root / "vectors.npy")

    # Configure UMAP using values from config. Cast types defensively.
    reducer = umap.UMAP(
        n_neighbors=int(cfg["umap"]["n_neighbors"]),
        min_dist=float(cfg["umap"]["min_dist"]),
        metric="cosine",
        random_state=42,
    )

    # Fit UMAP and transform the vectors to 2D
    latent_2d = reducer.fit_transform(vectors)

    # Persist the 2D latent coordinates for downstream visualization
    np.save(emb_root / "latent_2d.npy", latent_2d)
    print("Saved:", (emb_root / "latent_2d.npy").resolve(), latent_2d.shape)


if __name__ == "__main__":
    main()
