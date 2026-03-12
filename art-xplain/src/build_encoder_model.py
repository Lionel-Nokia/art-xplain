"""Build and save a style encoder model.

This module exposes `build_style_encoder_model` which constructs a Keras model
that maps input images to L2-normalized embedding vectors, and a `main`
utility that loads configuration, builds the model, and saves it.
"""

from __future__ import annotations
import tensorflow as tf
from .utils import load_config, ensure_dir

def build_style_encoder_model(img_size: int, embed_dim: int, backbone: str, freeze_backbone: bool):
    """Create a Keras encoder that outputs L2-normalized embeddings.

    Args:
        img_size: Size (height and width) of the input images in pixels.
        embed_dim: Dimensionality of the output embedding vector.
        backbone: Name of the backbone architecture to use (currently only
            "EfficientNetV2B0" is supported).
        freeze_backbone: If True, the backbone weights are frozen (not trainable).

    Returns:
        A compiled Keras `Model` that maps an RGB image to an L2-normalized vector.
    """

    # Define the input tensor shape (H, W, C)
    input_shape = (img_size, img_size, 3)

    # Select and configure the backbone model + corresponding preprocessing
    if backbone == "EfficientNetV2B0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        # Use the EfficientNetV2 preprocessing function to scale inputs correctly
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        raise ValueError(f"Backbone non supporté: {backbone}")

    # Enable or disable training for the backbone depending on config
    base_model.trainable = not freeze_backbone

    # Build the encoder head on top of the backbone
    inputs = tf.keras.Input(shape=input_shape)
    # Preprocess inputs (scales pixels to backbone's expected range)
    x = preprocess(inputs)
    # Pass through convolutional backbone. Let Keras control training/inference mode.
    x = base_model(x)
    # Pool spatial dimensions to produce a single feature vector per image
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Project to the desired embedding dimensionality
    x = tf.keras.layers.Dense(embed_dim)(x)
    # L2-normalize embeddings so they lie on the unit hypersphere
    x = tf.keras.layers.UnitNormalization(axis=1, name="l2norm")(x)
    model = tf.keras.Model(inputs, x, name="styledna_encoder")
    return model

def main():
    """Load configuration, build the encoder and save it to disk.

    The configuration is expected to provide `model.img_size`, `model.embed_dim`,
    `model.backbone`, `model.freeze_backbone`, and `paths.models_root`.
    """

    cfg = load_config()
    # Parse configuration values (cast to the expected types)
    img_size = int(cfg["model"]["img_size"])
    embed_dim = int(cfg["model"]["embed_dim"])
    backbone = str(cfg["model"]["backbone"])
    # Note: depending on the config loader, this may already be a boolean
    freeze_backbone = bool(cfg["model"]["freeze_backbone"])

    # Build and display a summary of the model
    model = build_style_encoder_model(img_size, embed_dim, backbone, freeze_backbone)
    model.summary()

    # Ensure the models directory exists and save the encoder there
    models_root = ensure_dir(cfg["paths"]["models_root"])
    model.save(models_root / "encoder.keras")
    print("Saved:", (models_root / "encoder.keras").resolve())

if __name__ == "__main__":
    main()
