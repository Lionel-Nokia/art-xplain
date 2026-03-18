"""Nearest-neighbour retrieval wrapper for StyleDNA embeddings.

This small utility loads precomputed image embeddings and a saved encoder
model, provides a convenience method to compute an embedding for a query
image, and returns the top-k most similar images (by cosine similarity).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from .utils import load_config
from .gradcam_similarity import GradCamSimilarity


class StyleRetriever:
    """Load embeddings and an encoder to perform nearest-neighbour search.

    The retriever expects the `embeddings_root` to contain `vectors.npy`,
    `labels.npy`, `filenames.npy` and `classnames.npy`, and `models_root`
    to contain a saved Keras model at `encoder.keras`.
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        # Load configuration and derive paths
        cfg = load_config(config_path)
        emb_root = Path(cfg["paths"]["embeddings_root"])
        models_root = Path(cfg["paths"]["models_root"])
        self.img_size = int(cfg["model"]["img_size"])

        # Load precomputed data used for retrieval
        self.embeddings = np.load(emb_root / "vectors.npy")
        self.labels = np.load(emb_root / "labels.npy")
        self.filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)
        self.classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)

        # Load the encoder used to compute query embeddings
        self.encoder = tf.keras.models.load_model(
            models_root / "encoder.keras", compile=False, safe_mode=False
        )
        self.gradcam = None
        self._gradcam_init_error = None

    def _load_image(self, image_path: str | Path) -> tf.Tensor:
        """Load an image from disk and return a batch tensor.

        The function uses Keras utilities to load and resize the image, then
        converts it to an array and adds a batch dimension so it can be
        directly fed into the encoder model.
        """

        img = tf.keras.utils.load_img(image_path, target_size=(self.img_size, self.img_size))
        arr = tf.keras.utils.img_to_array(img)
        return tf.expand_dims(arr, 0)

    def compute_query_embedding(self, image_path: str | Path) -> np.ndarray:
        """Compute the encoder embedding for a single query image.

        Returns a NumPy array of shape `(1, D)` where `D` is the embedding
        dimensionality.
        """

        img = self._load_image(image_path)
        emb = self.encoder(img, training=False).numpy()
        return emb  # (1, D)

    def top_k_similar(self, image_path: str | Path, k: int = 3):
        """Return a list of the top-k most similar images to the query.

        Each result is a dict containing `filepath`, `similarity` (cosine),
        `label_idx`, and the human-readable `style` computed from
        `classnames`.
        """

        query_emb = self.compute_query_embedding(image_path)

        # Compute cosine similarity between the query and all database vectors
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        # Get indices of top-k similarities (descending order)
        idxs = np.argsort(sims)[::-1][:k]

        out = []
        for i in idxs:
            out.append({
                "filepath": str(self.filenames[i]),
                "similarity": float(sims[i]),
                "label_idx": int(self.labels[i]),
                "style": str(self.classnames[self.labels[i]]),
            })
        return out

    def explain_similarity(self, query_path: str | Path, candidate_path: str | Path) -> dict:
        """Compute Grad-CAM overlays for a query/candidate pair."""
        if self.gradcam is None:
            try:
                self.gradcam = GradCamSimilarity(self.encoder, img_size=self.img_size)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._gradcam_init_error = exc
                raise RuntimeError(
                    f"Grad-CAM indisponible: {exc}. Le top-k reste utilisable."
                ) from exc
        return self.gradcam.explain_similarity(query_path, candidate_path)
