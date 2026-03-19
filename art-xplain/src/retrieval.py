"""Nearest-neighbour retrieval wrapper for StyleDNA embeddings."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from .utils import load_config


class StyleRetriever:
    """Load embeddings and an encoder to perform nearest-neighbour search."""

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        cfg = load_config(config_path)
        emb_root = Path(cfg["paths"]["embeddings_root"])
        models_root = Path(cfg["paths"]["models_root"])
        self.img_size = int(cfg["model"]["img_size"])

        self.embeddings = np.load(emb_root / "vectors.npy")
        self.labels = np.load(emb_root / "labels.npy")
        self.filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)
        self.classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)

        self.encoder = tf.keras.models.load_model(
            models_root / "encoder.keras",
            compile=False,
            safe_mode=False,
        )

        self.gradcam = None
        self._gradcam_init_error = None

    def _load_image(self, image_path: str | Path) -> tf.Tensor:
        img = tf.keras.utils.load_img(
            image_path,
            target_size=(self.img_size, self.img_size),
        )
        arr = tf.keras.utils.img_to_array(img)
        return tf.expand_dims(arr, 0)

    def compute_query_embedding(self, image_path: str | Path) -> np.ndarray:
        img = self._load_image(image_path)
        emb = self.encoder(img, training=False).numpy()
        return emb

    def top_k_similar(self, image_path: str | Path, k: int = 3):
        query_emb = self.compute_query_embedding(image_path)
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        idxs = np.argsort(sims)[::-1][:k]

        out = []
        for i in idxs:
            out.append({
                "filepath": str(self.filenames[i]),
                "similarity": float(sims[i]),
                "label_idx": int(self.labels[i]),
                "style": str(self.classnames[int(self.labels[i])]),
            })
        return out

    def explain_similarity(self, query_path: str | Path, candidate_path: str | Path) -> dict:
        if self.gradcam is None:
            try:
                from .gradcam_similarity import GradCamSimilarity
                self.gradcam = GradCamSimilarity(self.encoder, img_size=self.img_size)
            except Exception as exc:
                self._gradcam_init_error = exc
                raise RuntimeError(
                    f"Grad-CAM indisponible: {exc}. Le top-k reste utilisable."
                ) from exc

        return self.gradcam.explain_similarity(query_path, candidate_path)
