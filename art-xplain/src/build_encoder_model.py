"""Build and save a style encoder model.

This module exposes `build_style_encoder_model` which constructs a Keras model
that maps input images to L2-normalized embedding vectors, and a `main`
utility that loads configuration, builds the model, and saves it.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf
from .utils import load_config, ensure_dir


def print_step(step_number: int, title: str) -> None:
    separator = "=" * 72
    print(f"\n{separator}")
    print(f"build_encoder_model: {step_number} - {title}")
    print(separator)


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
        backbone: Name of the backbone architecture to use.
            Supported values include:
            - "EfficientNetV2-S"
            - "EfficientNetV2-M"
            - legacy aliases such as "EfficientNetV2B0"
        freeze_backbone: If True, the backbone weights are frozen (not trainable).

    Returns:
        A compiled Keras `Model` that maps an RGB image to an L2-normalized vector.
    """

    # Define the input tensor shape (H, W, C)
    input_shape = (img_size, img_size, 3)

    # Select and configure the backbone model + corresponding preprocessing
    backbone_aliases = {
        "EfficientNetV2-S": tf.keras.applications.EfficientNetV2S,
        "EfficientNetV2S": tf.keras.applications.EfficientNetV2S,
        "EfficientNetV2-M": tf.keras.applications.EfficientNetV2M,
        "EfficientNetV2M": tf.keras.applications.EfficientNetV2M,
        # Compatibilite ascendante avec l'ancienne config du projet.
        "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
    }

    backbone_builder = backbone_aliases.get(backbone)
    if backbone_builder is None:
        supported = ", ".join(backbone_aliases.keys())
        raise ValueError(f"Backbone non supporté: {backbone}. Valeurs possibles: {supported}")

    base_model = backbone_builder(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # Use the EfficientNetV2 preprocessing function to scale inputs correctly
    preprocess = tf.keras.applications.efficientnet_v2.preprocess_input

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
    model = tf.keras.Model(inputs, x, name="artxplain_encoder")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the encoder model from the prepared Keras dataset."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration YAML.",
    )
    return parser.parse_args()


def _get_backbone_from_encoder(encoder_model: tf.keras.Model) -> tf.keras.Model:
    for layer in reversed(encoder_model.layers):
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Backbone introuvable dans l'encodeur.")


def _set_finetune_layers(backbone_model: tf.keras.Model, n_last_layers: int) -> None:
    backbone_model.trainable = True
    if n_last_layers <= 0:
        for layer in backbone_model.layers:
            layer.trainable = False
        return

    split_idx = max(0, len(backbone_model.layers) - n_last_layers)
    for i, layer in enumerate(backbone_model.layers):
        layer.trainable = i >= split_idx


def _build_classifier(encoder: tf.keras.Model, img_size: int, n_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = encoder(inputs)
    logits = tf.keras.layers.Dense(n_classes, activation="softmax", name="style_head")(x)
    return tf.keras.Model(inputs, logits, name="styledna_classifier")


def train_encoder_model(config_path: str | Path = "config/config.yaml") -> None:
    print_step(1, "Chargement de la configuration")
    cfg = load_config(config_path)

    keras_root = Path(cfg["paths"]["keras_root"]).expanduser()
    models_root = Path(cfg["paths"]["models_root"]).expanduser()

    if not keras_root.is_absolute():
        keras_root = (Path.cwd() / keras_root).resolve()
    if not models_root.is_absolute():
        models_root = (Path.cwd() / models_root).resolve()

    print("keras_root :", keras_root)
    print("models_root:", models_root)

    print_step(2, "Verification des donnees d'entrainement")
    train_dir = keras_root / "train"
    val_dir = keras_root / "val"
    print("train_dir exists:", train_dir.exists())
    print("val_dir exists:", val_dir.exists())
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Dossiers train/val introuvables. Lance d'abord build_dataset_from_csv.")

    print_step(3, "Lecture des hyperparametres")
    img_size = int(cfg["model"]["img_size"])
    embed_dim = int(cfg["model"]["embed_dim"])
    backbone = str(cfg["model"]["backbone"])
    freeze_backbone = bool(cfg["model"].get("freeze_backbone", True))

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs_head = int(train_cfg.get("epochs_head", 8))
    epochs_finetune = int(train_cfg.get("epochs_finetune", 10))
    lr_head = float(train_cfg.get("lr_head", 1e-3))
    lr_finetune = float(train_cfg.get("lr_finetune", 1e-5))
    finetune_last_layers = int(train_cfg.get("finetune_last_layers", 60))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 4))

    data_cfg = cfg.get("dataset", {})
    max_images = int(data_cfg.get("max_images", 5000))
    keep_top_styles = int(data_cfg.get("keep_top_styles", 5))
    min_images_per_style = int(data_cfg.get("min_images_per_style", 200))

    print("-- Parametres dataset --")
    print("max_images:", max_images)
    print("keep_top_styles:", keep_top_styles)
    print("min_images_per_style:", min_images_per_style)

    print("\n-- Hyperparametres d'entrainement --")
    print("img_size:", img_size)
    print("embed_dim:", embed_dim)
    print("backbone:", backbone)
    print("freeze_backbone:", freeze_backbone)
    print("batch_size:", batch_size)
    print("epochs_head:", epochs_head, "epochs_finetune:", epochs_finetune)
    print("lr_head:", lr_head)
    print("lr_finetune:", lr_finetune)
    print("finetune_last_layers:", finetune_last_layers)
    print("early_stopping_patience:", early_stopping_patience)

    print_step(4, "Creation des datasets TensorFlow")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_names=train_ds.class_names,
    )

    class_names = train_ds.class_names
    n_classes = len(class_names)
    print("classes:", class_names)
    print("n_classes:", n_classes)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    print_step(5, "Construction du modele")
    encoder = build_style_encoder_model(
        img_size=img_size,
        embed_dim=embed_dim,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
    )
    encoder.summary()
    classifier = _build_classifier(encoder, img_size, n_classes)
    classifier.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    print_step(6, "Entrainement de la tete de classification")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_head),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    print(f"Phase 1/2: entrainement de la tete ({epochs_head} epochs)")
    classifier.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks,
    )

    print_step(7, "Fine-tuning du backbone")
    if epochs_finetune > 0:
        backbone_model = _get_backbone_from_encoder(encoder)
        _set_finetune_layers(backbone_model, finetune_last_layers)
        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_finetune),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        print(
            f"Phase 2/2: fine-tuning ({epochs_finetune} epochs, "
            f"{finetune_last_layers} dernieres couches)"
        )
        classifier.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_finetune,
            callbacks=callbacks,
        )
    else:
        print("Fine-tuning ignore (epochs_finetune=0)")

    print_step(8, "Sauvegarde du modele")
    ensure_dir(models_root)
    output_path = models_root / "encoder.keras"
    encoder.save(output_path)
    print("Saved encoder:", output_path.resolve())

    print_step(9, "Resume final")
    print("Saved encoder:", output_path.resolve())
    print("Classes:", class_names)
    print("Nombre de classes:", n_classes)


def main() -> None:
    args = parse_args()
    try:
        train_encoder_model(args.config)
    except KeyboardInterrupt:
        print("\n" + "=" * 72)
        print("Execution interrompue par l'utilisateur (Ctrl+C).")
        print("Aucun nettoyage supplementaire n'a ete necessaire.")
        print("=" * 72)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
