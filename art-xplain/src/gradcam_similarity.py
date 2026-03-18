"""Grad-CAM style explanations for embedding-based image similarity."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class GradCamSimilarity:
    """Compute Grad-CAM overlays for query/candidate similarity."""

    def __init__(self, encoder: tf.keras.Model, img_size: int):
        self.encoder = encoder
        self.img_size = int(img_size)

        self._backbone = self._find_backbone_model()
        self._target_layer_name = f"{self._backbone.name}.output"

        # Rebuild the encoder forward path layer-by-layer to capture a
        # backbone feature map tensor that remains connected to the final
        # embedding output used in the similarity score.
        x = self.encoder.inputs[0]
        target_tensor = None
        for layer in self.encoder.layers[1:]:
            x = layer(x)
            if layer.name == self._backbone.name:
                target_tensor = x

        if target_tensor is None:
            raise ValueError("Impossible de récupérer la sortie du backbone pour Grad-CAM.")

        self.grad_model = tf.keras.Model(
            self.encoder.inputs,
            [target_tensor, x],
            name="gradcam_similarity_model",
        )

    def _find_backbone_model(self) -> tf.keras.Model:
        for layer in reversed(self.encoder.layers):
            if isinstance(layer, tf.keras.Model):
                return layer
        raise ValueError("Impossible de trouver le backbone dans l'encodeur.")

    def _load_image(self, image_path: str | Path) -> tuple[tf.Tensor, np.ndarray]:
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_size, self.img_size))
        arr = tf.keras.utils.img_to_array(img)  # float32 [0..255]
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        batch = tf.expand_dims(arr, 0)
        return batch, arr_uint8

    @staticmethod
    def _cosine_sim(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        a = tf.math.l2_normalize(a, axis=1)
        b = tf.math.l2_normalize(b, axis=1)
        return tf.reduce_sum(a * b, axis=1)

    def _cam_from_input(self, image_batch: tf.Tensor, fixed_embedding: tf.Tensor) -> tuple[np.ndarray, float]:
        with tf.GradientTape() as tape:
            conv_maps, emb = self.grad_model(image_batch, training=False)
            score = self._cosine_sim(emb, tf.stop_gradient(fixed_embedding))

        grads = tape.gradient(score, conv_maps)
        if grads is None:
            raise RuntimeError("Gradient indisponible pour Grad-CAM.")

        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_maps, axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam[0]

        max_val = tf.reduce_max(cam)
        if float(max_val.numpy()) > 0.0:
            cam = cam / max_val

        cam_np = cam.numpy()
        cam_np = cv2.resize(cam_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        cam_np = np.clip(cam_np, 0.0, 1.0)
        return cam_np, float(score.numpy()[0])

    @staticmethod
    def _overlay_heatmap(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        heat = np.uint8(255 * cam)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(image_uint8, 1.0 - alpha, heat, alpha, 0)
        return blended

    def explain_similarity(self, query_path: str | Path, candidate_path: str | Path) -> dict:
        """Return CAM overlays for query and candidate images."""

        query_batch, query_img = self._load_image(query_path)
        candidate_batch, candidate_img = self._load_image(candidate_path)

        query_emb = self.encoder(query_batch, training=False)
        candidate_emb = self.encoder(candidate_batch, training=False)

        query_cam, similarity = self._cam_from_input(query_batch, candidate_emb)
        candidate_cam, _ = self._cam_from_input(candidate_batch, query_emb)

        return {
            "similarity": similarity,
            "target_layer": self._target_layer_name,
            "query_cam": query_cam,
            "candidate_cam": candidate_cam,
            "query_overlay": self._overlay_heatmap(query_img, query_cam),
            "candidate_overlay": self._overlay_heatmap(candidate_img, candidate_cam),
        }
