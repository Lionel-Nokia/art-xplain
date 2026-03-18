"""Build and save a style encoder model.

This module exposes `build_style_encoder_model` which constructs a Keras model
that maps input images to L2-normalized embedding vectors, and a `main`
utility that loads configuration, builds the model, and saves it.
"""

from __future__ import annotations
import tensorflow as tf
from .utils import load_config, ensure_dir


def limit_max_files(df, max_files=None, shuffle=True, random_state=42):
    """
    Limite le nombre de lignes (fichiers) dans un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les fichiers.
        max_files (int | None): nombre maximum de fichiers à garder.
        shuffle (bool): mélanger avant de couper pour éviter les biais.
        random_state (int): seed pour reproductibilité.

    Returns:
        pd.DataFrame
    """

    if max_files is None:
        return df

    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    if len(df) > max_files:
        df = df.head(max_files)

    return df


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
