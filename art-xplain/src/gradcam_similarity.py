"""Grad-CAM++ explanations for embedding-based image similarity."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class GradCamPlusPlusSimilarity:
    """Compute Grad-CAM++ overlays for query/candidate similarity."""

    CONV_LAYER_TYPES = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    def __init__(
        self,
        encoder: tf.keras.Model,
        img_size: int,
        target_layer_name: str | None = None,
    ):
        self.encoder = encoder
        self.img_size = int(img_size)

        self._backbone = self._find_backbone_model()
        self._post_backbone_layers = self._find_post_backbone_layers()

        available = self.list_target_layers(self.encoder)
        if not available:
            raise ValueError("Aucune couche convolutionnelle compatible pour Grad-CAM++.")

        self._target_layer_name = target_layer_name or available[-1]
        target_layer = self._resolve_target_layer(self._target_layer_name)

        backbone_input = self._backbone.input
        backbone_output = self._backbone.output
        feature_extractor = tf.keras.Model(
            backbone_input,
            [target_layer.output, backbone_output],
            name=f"gradcampp_backbone_{target_layer.name}",
        )

        image_input = self.encoder.inputs[0]
        conv_maps, x = feature_extractor(image_input)
        for layer in self._post_backbone_layers:
            x = layer(x)

        self.grad_model = tf.keras.Model(
            image_input,
            [conv_maps, x],
            name=f"gradcampp_similarity_{target_layer.name}",
        )

    @staticmethod
    def list_target_layers(encoder: tf.keras.Model) -> list[str]:
        """Return selectable convolutional layers from the encoder backbone."""
        backbone = None
        for layer in reversed(encoder.layers):
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break

        if backbone is None:
            return []

        def is_4d_feature_map(layer: tf.keras.layers.Layer) -> bool:
            shape = getattr(layer.output, "shape", None)
            return shape is not None and len(shape) == 4

        return [
            layer.name
            for layer in backbone.layers
            if isinstance(layer, GradCamPlusPlusSimilarity.CONV_LAYER_TYPES)
            and is_4d_feature_map(layer)
        ]

    def _find_backbone_model(self) -> tf.keras.Model:
        for layer in reversed(self.encoder.layers):
            if isinstance(layer, tf.keras.Model):
                return layer
        raise ValueError("Impossible de trouver le backbone dans l'encodeur.")

    def _find_post_backbone_layers(self) -> list[tf.keras.layers.Layer]:
        for idx, layer in enumerate(self.encoder.layers):
            if layer is self._backbone:
                return list(self.encoder.layers[idx + 1 :])
        raise ValueError("Impossible de reconstruire la tête de l'encodeur.")

    def _resolve_target_layer(self, target_layer_name: str) -> tf.keras.layers.Layer:
        try:
            return self._backbone.get_layer(target_layer_name)
        except ValueError as exc:
            available = ", ".join(self.list_target_layers(self.encoder))
            raise ValueError(
                f"Couche Grad-CAM++ introuvable: {target_layer_name}. "
                f"Couches disponibles: {available}"
            ) from exc

    def _load_image(self, image_path: str | Path) -> tuple[tf.Tensor, np.ndarray]:
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_size, self.img_size))
        arr = tf.keras.utils.img_to_array(img)
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        batch = tf.expand_dims(arr, 0)
        return batch, arr_uint8

    @staticmethod
    def _cosine_sim(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        a = tf.math.l2_normalize(a, axis=1)
        b = tf.math.l2_normalize(b, axis=1)
        return tf.reduce_sum(a * b, axis=1)

    def _cam_from_input(
        self,
        image_batch: tf.Tensor,
        fixed_embedding: tf.Tensor,
    ) -> tuple[np.ndarray, float]:
        with tf.GradientTape() as tape:
            conv_maps, emb = self.grad_model(image_batch, training=False)
            score = self._cosine_sim(emb, tf.stop_gradient(fixed_embedding))

        grads = tape.gradient(score, conv_maps)
        if grads is None:
            raise RuntimeError("Gradient indisponible pour Grad-CAM++.")

        positive_grads = tf.nn.relu(grads)
        grads_2 = tf.square(grads)
        grads_3 = grads_2 * grads

        spatial_sum = tf.reduce_sum(conv_maps, axis=(1, 2), keepdims=True)
        alpha_denom = (2.0 * grads_2) + (grads_3 * spatial_sum)
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
        alphas = grads_2 / alpha_denom

        alpha_norm = tf.reduce_sum(alphas, axis=(1, 2), keepdims=True)
        alpha_norm = tf.where(alpha_norm != 0.0, alpha_norm, tf.ones_like(alpha_norm))
        alphas = alphas / alpha_norm

        weights = tf.reduce_sum(alphas * positive_grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_maps, axis=-1)
        cam = tf.nn.relu(cam)[0]

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
        """Return Grad-CAM++ overlays for query and candidate images."""
        query_batch, query_img = self._load_image(query_path)
        candidate_batch, candidate_img = self._load_image(candidate_path)

        query_emb = self.encoder(query_batch, training=False)
        candidate_emb = self.encoder(candidate_batch, training=False)

        query_cam, similarity = self._cam_from_input(query_batch, candidate_emb)
        candidate_cam, _ = self._cam_from_input(candidate_batch, query_emb)

        return {
            "method": "Grad-CAM++",
            "similarity": similarity,
            "target_layer": self._target_layer_name,
            "query_cam": query_cam,
            "candidate_cam": candidate_cam,
            "query_overlay": self._overlay_heatmap(query_img, query_cam),
            "candidate_overlay": self._overlay_heatmap(candidate_img, candidate_cam),
        }
