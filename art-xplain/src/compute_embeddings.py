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
# Active l'évaluation différée des annotations de type.
# Cela permet d'utiliser une syntaxe moderne de type hints
# tout en gardant une bonne compatibilité et une résolution plus souple.

from pathlib import Path
# `Path` fournit une API orientée objet pour manipuler les chemins de fichiers.
# C'est plus robuste et lisible que de concaténer des chaînes manuellement.
import hashlib
# `hashlib` sert ici à calculer des empreintes (SHA1) des tableaux exportés.
# Ces empreintes sont utiles pour vérifier qu'un export n'a pas changé silencieusement.
import json
# `json` permet d'écrire un fichier manifeste lisible par un humain
# et facilement réutilisable par d'autres scripts.
import sys
# `sys` est utilisé en fin de script pour écrire proprement les erreurs sur stderr.

import numpy as np
# NumPy est la bibliothèque centrale pour manipuler les embeddings,
# labels et métadonnées sous forme de tableaux.
import tensorflow as tf
# TensorFlow / Keras est utilisé pour :
# - charger le modèle encodeur,
# - préparer les images,
# - produire les vecteurs de représentation (embeddings).

from .utils import ensure_dir, load_config, relativize_project_path, resolve_project_path
# Fonctions utilitaires internes du projet :
# - `load_config` charge la configuration YAML,
# - `ensure_dir` crée le dossier cible si besoin,
# - `resolve_project_path` convertit un chemin de config en chemin projet fiable.


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# Ensemble des extensions considérées comme des images valides pour ce pipeline.
# Le choix d'un `set` est pratique pour des tests d'appartenance rapides et lisibles.


def _sha1_of_array(arr: np.ndarray) -> str:
    """
    Calcule une empreinte SHA1 déterministe à partir d'un tableau NumPy.

    L'objectif n'est pas la sécurité cryptographique, mais la traçabilité :
    si le contenu du tableau change, son hash changera aussi.
    """
    h = hashlib.sha1()
    # Crée un objet hash SHA1 incrémental.

    h.update(np.ascontiguousarray(arr).view(np.uint8))
    # `np.ascontiguousarray` garantit un stockage mémoire contigu.
    # C'est important pour obtenir une représentation binaire stable.
    # `.view(np.uint8)` revoit les mêmes octets comme une suite d'entiers 8 bits,
    # ce qui permet de hasher le contenu brut du tableau.

    return h.hexdigest()
    # Renvoie l'empreinte sous forme hexadécimale lisible.


def _list_class_dirs(root: Path) -> list[Path]:
    """
    Retourne la liste triée des sous-dossiers de classes sous `root`.

    Dans un dataset organisé "à la Keras", chaque sous-dossier représente
    généralement une classe, ici un style artistique.
    """
    return sorted([p for p in root.iterdir() if p.is_dir()])
    # `root.iterdir()` parcourt le contenu immédiat du dossier.
    # On garde uniquement les dossiers, puis on trie pour obtenir
    # un ordre stable et reproductible entre deux exécutions.


def _list_images(folder: Path) -> list[Path]:
    """
    Liste récursivement toutes les images valides présentes dans un dossier.
    """
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    # `rglob("*")` parcourt récursivement tous les fichiers et sous-dossiers.
    # On filtre ensuite :
    # - `p.is_file()` pour ignorer les dossiers,
    # - `p.suffix.lower() in IMG_EXTS` pour ne garder que les formats image attendus.
    # Le `sorted(...)` garantit là encore un ordre stable.


def _load_image_for_encoder(image_path: str | Path, img_size: int) -> tf.Tensor:
    """
    Charge une image et la convertit dans le format attendu par l'encodeur.

    Cette fonction reproduit volontairement la même logique que `retrieval.py`
    pour éviter tout décalage entre :
    - la manière dont on calcule les embeddings de la galerie,
    - la manière dont on calcule l'embedding d'une image requête.
    """
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    # Charge l'image depuis le disque et la redimensionne directement
    # à la taille attendue par le modèle.

    arr = tf.keras.utils.img_to_array(img)
    # Convertit l'image PIL en tenseur / tableau numérique exploitable par Keras.

    return tf.expand_dims(arr, 0)
    # Ajoute une dimension batch en première position.
    # Le modèle attend typiquement une forme `(batch, height, width, channels)`.
    # Une image seule devient donc ici un batch de taille 1.


def _compute_embedding(
    encoder: tf.keras.Model,
    image_path: str | Path,
    img_size: int,
) -> np.ndarray:
    """
    Calcule l'embedding d'une image unique via l'encodeur fourni.
    """
    batch = _load_image_for_encoder(image_path, img_size)
    # Prépare l'image au format d'entrée du modèle.

    emb = encoder(batch, training=False).numpy()
    # Exécute l'encodeur en mode inférence (`training=False`),
    # puis convertit la sortie TensorFlow en tableau NumPy.

    if emb.ndim != 2 or emb.shape[0] != 1:
        raise ValueError(
            f"Sortie encoder invalide pour {image_path}: shape={emb.shape}, attendu=(1, D)"
        )
        # On attend exactement une matrice 2D dont la première dimension vaut 1 :
        # - 2D car il y a un batch,
        # - batch=1 car on traite une seule image à la fois.
        # Si ce contrat n'est pas respecté, mieux vaut échouer tôt avec un message clair.

    return emb[0].astype(np.float32)
    # On enlève la dimension batch pour garder seulement le vecteur d'embedding.
    # `float32` est un bon compromis :
    # - assez précis pour les embeddings,
    # - plus compact que float64 en mémoire et sur disque.


def _resolve_dataset_sources(cfg: dict) -> tuple[list[Path], str]:
    """
    Détermine un ou plusieurs dossiers source pour calculer les embeddings.

    Comportement :
    - si `paths.dataset_root` est défini, il reste prioritaire et unique ;
    - sinon, on lit `dataset.embedding_splits` dans la config ;
    - si ce paramètre est absent, on utilise par défaut `["train", "val"]`.

    Cela permet de choisir explicitement entre :
    - `train + val`
    - `train + val + test`
    """
    paths_cfg = cfg.get("paths", {})

    if "dataset_root" in paths_cfg:
        root = resolve_project_path(paths_cfg["dataset_root"])
        return [root], "dataset_root"
        # `dataset_root` garde la priorité absolue pour préserver
        # un mode de fonctionnement simple et explicite.

    if "keras_root" not in paths_cfg:
        raise KeyError(
            "Ni 'paths.dataset_root' ni 'paths.keras_root' n'existent dans config.yaml"
        )

    dataset_cfg = cfg.get("dataset", {})
    raw_splits = dataset_cfg.get("embedding_splits", ["train", "val"])
    # Par défaut, on construit donc la galerie à partir de `train + val`,
    # ce qui est généralement plus pertinent pour une app de retrieval
    # que d'utiliser `test` seul.

    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError(
            "`dataset.embedding_splits` doit être une liste YAML non vide, "
            "par exemple ['train', 'val']"
        )

    allowed_splits = {"train", "val", "test"}
    normalized_splits: list[str] = []

    for split in raw_splits:
        split_name = str(split).strip().lower()
        if split_name not in allowed_splits:
            raise ValueError(
                "`dataset.embedding_splits` ne peut contenir que "
                "'train', 'val' ou 'test'."
            )
        if split_name not in normalized_splits:
            normalized_splits.append(split_name)
            # On déduplique tout en conservant l'ordre défini dans la config.

    keras_root = resolve_project_path(paths_cfg["keras_root"])
    selected_roots: list[Path] = []
    missing_splits: list[str] = []

    for split_name in normalized_splits:
        split_root = keras_root / split_name
        if split_root.exists() and split_root.is_dir():
            selected_roots.append(split_root)
        else:
            missing_splits.append(split_name)

    if missing_splits:
        raise FileNotFoundError(
            "Certains splits demandés pour les embeddings sont introuvables : "
            f"{missing_splits} sous {keras_root}"
        )

    if not selected_roots:
        raise FileNotFoundError(
            f"Aucun split exploitable trouvé sous {keras_root} "
            f"pour dataset.embedding_splits={normalized_splits}"
        )

    return selected_roots, " + ".join(f"keras_root/{split}" for split in normalized_splits)


def _collect_images_by_class(dataset_roots: list[Path]) -> dict[str, list[Path]]:
    """
    Regroupe les images par nom de classe à travers plusieurs splits.

    Exemple :
    - `train/impressionism/...`
    - `val/impressionism/...`

    seront fusionnés sous une seule classe `impressionism`.
    """
    images_by_class: dict[str, list[Path]] = {}

    for root in dataset_roots:
        for class_dir in _list_class_dirs(root):
            class_name = class_dir.name
            image_files = _list_images(class_dir)
            images_by_class.setdefault(class_name, []).extend(image_files)

    for class_name in images_by_class:
        images_by_class[class_name] = sorted(images_by_class[class_name])
        # On retrie chaque classe pour conserver un ordre stable,
        # même quand les images proviennent de plusieurs splits.

    return dict(sorted(images_by_class.items()))
    # Le tri final sur les noms de classes rend aussi l'attribution
    # des labels numériques stable d'une exécution à l'autre.


def main():
    """
    Point d'entrée principal du script de calcul d'embeddings.

    Le flux global est le suivant :
    1. charger la configuration,
    2. résoudre les chemins utiles,
    3. charger l'encodeur,
    4. parcourir le dataset classe par classe,
    5. calculer un embedding par image,
    6. sauvegarder tableaux + bundle + manifeste.
    """
    cfg = load_config()
    # Charge la configuration du projet depuis le fichier prévu par l'application.

    dataset_roots, dataset_source = _resolve_dataset_sources(cfg)
    embeddings_root = resolve_project_path(cfg["paths"]["embeddings_root"])
    models_root = resolve_project_path(cfg["paths"]["models_root"])
    ensure_dir(embeddings_root)
    # On résout les chemins de travail, puis on s'assure que le dossier de sortie existe.
    # `ensure_dir` évite d'échouer au moment de l'écriture des fichiers exportés.

    img_size = int(cfg["model"]["img_size"])
    encoder_path = models_root / "encoder.keras"
    # `img_size` doit rester cohérent avec la taille d'entrée attendue par le modèle.
    # `encoder_path` pointe vers le modèle d'encodage entraîné.

    for dataset_root in dataset_roots:
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dossier source introuvable: {dataset_root} "
                f"(résolu depuis {dataset_source})"
            )
            # Erreur explicite si l'une des sources d'images n'existe pas.

    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder.keras introuvable: {encoder_path}")
        # Erreur explicite si le modèle attendu n'est pas disponible.

    print("Sources images  :")
    for dataset_root in dataset_roots:
        print(f" - {dataset_root.resolve()}")
    print(f"Mode sélection  : {dataset_source}")
    print(f"Embeddings root : {embeddings_root.resolve()}")
    print(f"Encoder path    : {encoder_path.resolve()}")
    print(f"Image size      : {img_size}")

    encoder = tf.keras.models.load_model(
        encoder_path,
        compile=False,
        safe_mode=False,
    )
    # Charge le modèle Keras depuis le disque.
    # `compile=False` est suffisant ici car on ne réentraîne pas le modèle.
    # On veut seulement l'utiliser en inférence.

    images_by_class = _collect_images_by_class(dataset_roots)
    if not images_by_class:
        roots_str = ", ".join(str(p) for p in dataset_roots)
        raise ValueError(f"Aucun dossier de classe trouvé dans les sources: {roots_str}")
        # Sans dossier de classe, le script ne peut pas construire
        # de labels cohérents pour les embeddings.

    vectors_list: list[np.ndarray] = []
    labels_list: list[int] = []
    filenames_list: list[str] = []
    classnames_list: list[str] = []
    skipped_files: list[dict[str, str]] = []
    # Ces listes servent de tampons Python pendant le parcours du dataset.
    # On accumule d'abord les résultats au fil de l'eau,
    # puis on convertit à la fin vers des tableaux NumPy homogènes.

    print(f"Nombre de styles détectés : {len(images_by_class)}")

    for class_name, image_files in images_by_class.items():
        if not image_files:
            print(f"[WARN] Aucun fichier image pour la classe {class_name}")
            continue

        class_idx = len(classnames_list)
        classnames_list.append(class_name)
        # Le nom du dossier devient le nom de classe.
        # Son index dans `classnames_list` devient l'identifiant numérique de cette classe.
        # Ce couplage est simple, lisible et reproductible grâce au tri préalable des classes.

        print(f"-> {class_name}: {len(image_files)} image(s)")

        valid_count = 0
        for img_path in image_files:
            try:
                vec = _compute_embedding(encoder, img_path, img_size)
                vectors_list.append(vec)
                labels_list.append(class_idx)
                filenames_list.append(relativize_project_path(img_path))
                valid_count += 1
                # Pour chaque image valide, on stocke :
                # - son vecteur,
                # - son label numérique,
                # - son chemin, désormais relatif au projet si possible,
                #   pour rendre l'export portable quand on déplace le repo.
            except Exception as exc:
                skipped_files.append({
                    "file": str(img_path),
                    "error": str(exc),
                })
                print(f"[WARN] Fichier ignoré: {img_path} -> {exc}")
                # Une image défectueuse ne bloque pas tout le pipeline.
                # On la loggue, on garde sa trace dans le manifeste,
                # puis on continue avec les autres fichiers.

        print(f"   {valid_count} embedding(s) calculé(s)")
        # Petit bilan par classe, utile pendant l'exécution pour vérifier
        # qu'aucune catégorie n'est massivement ignorée.

    if not vectors_list:
        raise ValueError("Aucun embedding n'a pu être calculé.")
        # Sécurité importante : si rien n'a été produit, on échoue clairement
        # plutôt que d'écrire des fichiers vides ou incohérents.

    vectors = np.stack(vectors_list).astype(np.float32)
    labels = np.asarray(labels_list, dtype=np.int64)
    filenames = np.asarray(filenames_list, dtype=object)
    classnames = np.asarray(classnames_list, dtype=object)
    # Conversion finale vers des tableaux NumPy :
    # - `vectors` devient une vraie matrice 2D `(N, D)`,
    # - `labels` contient les identifiants entiers de classes,
    # - `filenames` et `classnames` restent en `object` car ce sont des chaînes.

    n_samples = vectors.shape[0]
    # Nombre total d'images valides transformées en embeddings.

    if not (n_samples == len(labels) == len(filenames)):
        raise RuntimeError(
            "Incohérence de cardinalité après calcul des embeddings.\n"
            f" - vectors   : {n_samples}\n"
            f" - labels    : {len(labels)}\n"
            f" - filenames : {len(filenames)}"
        )
        # Contrôle de cohérence fondamental :
        # chaque embedding doit avoir exactement un label et un nom de fichier associé.

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
            # On vérifie que tous les labels pointent bien vers une classe existante.
            # Cela évite des erreurs plus tard dans les scripts de retrieval ou de visualisation.

    bundle_path = embeddings_root / "embeddings_bundle.npz"
    np.savez_compressed(
        bundle_path,
        vectors=vectors,
        labels=labels,
        filenames=filenames,
        classnames=classnames,
    )
    # `np.savez_compressed` crée un bundle unique compact.
    # C'est très pratique pour transporter ou charger toutes les données associées
    # à un export d'embeddings en une seule lecture.

    np.save(embeddings_root / "vectors.npy", vectors)
    np.save(embeddings_root / "labels.npy", labels)
    np.save(embeddings_root / "filenames.npy", filenames)
    np.save(embeddings_root / "classnames.npy", classnames)
    # On sauvegarde aussi les artefacts séparément.
    # Cela facilite le debug, l'inspection manuelle et la compatibilité
    # avec d'autres scripts du projet qui pourraient consommer un seul fichier précis.

    manifest = {
        "num_samples": int(n_samples),
        "embedding_dim": int(vectors.shape[1]),
        "num_classes": int(len(classnames)),
        "img_size": int(img_size),
        "dataset_roots": [str(path.resolve()) for path in dataset_roots],
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
    # Le manifeste résume le contenu de l'export et garde des informations de traçabilité :
    # dimensions, chemins source, empreintes, nombre de fichiers ignorés, etc.
    # Limiter `skipped_files` aux 200 premiers évite de produire un JSON énorme
    # si beaucoup d'images posent problème.

    manifest_path = embeddings_root / "embeddings_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # On écrit le manifeste en JSON lisible (`indent=2`).
    # `ensure_ascii=False` conserve correctement les accents si besoin.

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
        # On réécrit l'erreur sur la sortie d'erreur standard pour un diagnostic plus propre,
        # puis on la relance afin de conserver un code de sortie non nul.
        raise
