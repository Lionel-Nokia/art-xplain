"""Nearest-neighbour retrieval wrapper for StyleDNA embeddings."""

from __future__ import annotations
# Active l'évaluation différée des annotations de type.
# Cela permet d'utiliser des annotations modernes comme `dict[str, str]`
# tout en gardant un comportement souple au chargement du module.

from pathlib import Path
# `Path` fournit une manière plus robuste et lisible de manipuler des chemins
# que les simples chaînes de caractères.
import hashlib
# `hashlib` est utilisé ici pour calculer des empreintes SHA1 de fichiers.
# Le but n'est pas la sécurité, mais la détection fiable de deux fichiers
# qui ont exactement le même contenu binaire.

import numpy as np
# NumPy sert ici à :
# - charger les tableaux d'embeddings et métadonnées,
# - manipuler les indices des meilleurs voisins,
# - stocker certaines structures sous forme de arrays.
import tensorflow as tf
# TensorFlow / Keras est utilisé pour recharger l'encodeur
# et calculer l'embedding de l'image requête.
from sklearn.metrics.pairwise import cosine_similarity
# `cosine_similarity` compare le vecteur de la requête
# avec tous les embeddings déjà présents dans la galerie.
# C'est le cœur du moteur de nearest neighbours ici.

from .utils import load_config, resolve_project_path, resolve_stored_path
# Utilitaires internes :
# - `load_config` lit la configuration YAML,
# - `resolve_project_path` résout un chemin relatif au projet,
# - `resolve_stored_path` permet de relire proprement des chemins sauvegardés
#   dans les exports, qu'ils soient absolus ou relatifs.


class StyleRetriever:
    """
    Charge les embeddings de la galerie ainsi que l'encodeur,
    puis fournit des méthodes pour :
    - encoder une image requête,
    - retrouver les images les plus similaires,
    - produire une explication Grad-CAM optionnelle.

    Cette classe joue donc le rôle de couche "métier" entre :
    - les artefacts pré-calculés (`vectors.npy`, `filenames.npy`, etc.),
    - et l'interface utilisateur Streamlit.
    """

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        """
        Initialise le moteur de retrieval à partir de la configuration du projet.

        Au démarrage, on charge :
        - la configuration,
        - les embeddings et métadonnées de la galerie,
        - le modèle encodeur Keras.

        L'idée est que cet objet soit ensuite réutilisé pour plusieurs requêtes,
        plutôt que reconstruit à chaque recherche.
        """
        cfg = load_config(config_path)
        # Charge le fichier de configuration YAML.
        # `config_path` peut être relatif ou absolu.

        emb_root = resolve_project_path(cfg["paths"]["embeddings_root"])
        models_root = resolve_project_path(cfg["paths"]["models_root"])
        # Résout les dossiers d'artefacts du projet :
        # - `emb_root` contient les embeddings et métadonnées de la galerie,
        # - `models_root` contient le modèle encodeur entraîné.

        self.img_size = int(cfg["model"]["img_size"])
        # Taille d'entrée attendue par le modèle.
        # Toutes les images de requête seront redimensionnées à cette taille.

        self.embeddings = np.load(emb_root / "vectors.npy")
        # Matrice des embeddings de la galerie, de forme typique `(N, D)` :
        # - N = nombre d'images référencées dans la base,
        # - D = dimension du vecteur d'embedding.

        self.labels = np.load(emb_root / "labels.npy")
        # Label numérique associé à chaque embedding.
        # Il permet de relier un embedding à une classe/style.

        self.filenames = np.load(emb_root / "filenames.npy", allow_pickle=True)
        # Liste des chemins d'images tels qu'ils ont été sauvegardés à l'export.
        # Depuis la dernière évolution du projet, ces chemins peuvent être relatifs
        # à la racine du projet, ce qui rend l'export portable.

        self.classnames = np.load(emb_root / "classnames.npy", allow_pickle=True)
        # Tableau qui mappe un label numérique vers un nom de style lisible.

        self.resolved_filenames = np.asarray(
            [str(resolve_stored_path(fp)) for fp in self.filenames],
            dtype=object,
        )
        # On prépare dès l'initialisation une version "résolue" des chemins d'images.
        # Pourquoi ?
        # - `filenames.npy` peut contenir des chemins relatifs,
        # - mais l'application et Streamlit ont ensuite besoin de chemins réels exploitables.
        # Cette conversion en amont simplifie le reste du code.

        self.encoder = tf.keras.models.load_model(
            models_root / "encoder.keras",
            compile=False,
            safe_mode=False,
        )
        # Recharge l'encodeur entraîné.
        # `compile=False` suffit car on ne veut pas entraîner le modèle ici,
        # seulement l'utiliser en inférence.

        self.gradcam = None
        # L'objet Grad-CAM n'est pas créé immédiatement.
        # On applique ici une stratégie de "lazy loading" :
        # on ne le construira que si l'utilisateur demande réellement
        # une explication visuelle.

        self._gradcam_init_error = None
        # Permet de garder la trace d'une erreur éventuelle lors de l'initialisation
        # de Grad-CAM, afin de la remonter proprement.

        self._file_hash_cache: dict[str, str] = {}
        # Petit cache mémoire des empreintes SHA1 de fichiers.
        # Cela évite de recalculer plusieurs fois le hash d'un même fichier,
        # ce qui est utile lors des recherches répétées.

    def _sha1_of_file(self, file_path: str | Path) -> str:
        """
        Calcule l'empreinte SHA1 d'un fichier image.

        Cette méthode sert surtout à exclure les auto-matches :
        si l'image requête est une copie exacte d'une image déjà dans la galerie,
        leurs contenus binaires auront le même hash.
        """
        path = str(Path(file_path).resolve())
        # On normalise le chemin sous forme absolue pour éviter
        # de cacher plusieurs fois le même fichier sous des chemins différents.

        cached = self._file_hash_cache.get(path)
        if cached is not None:
            return cached
            # Si l'empreinte a déjà été calculée, on la réutilise immédiatement.

        h = hashlib.sha1()
        # Crée un objet hash incrémental.

        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        # Lecture par blocs de 1 Mo.
        # Cette approche évite de charger tout le fichier en mémoire d'un coup.

        digest = h.hexdigest()
        # Transforme le hash en chaîne hexadécimale lisible.

        self._file_hash_cache[path] = digest
        # Mémorise le résultat pour les appels suivants.

        return digest

    def _load_image(self, image_path: str | Path) -> tf.Tensor:
        """
        Charge une image requête et la convertit dans le format attendu par l'encodeur.
        """
        img = tf.keras.utils.load_img(
            image_path,
            target_size=(self.img_size, self.img_size),
        )
        # Charge l'image depuis le disque et la redimensionne immédiatement
        # à la taille d'entrée du modèle.

        arr = tf.keras.utils.img_to_array(img)
        # Convertit l'image en tenseur / tableau numérique exploitable par Keras.

        return tf.expand_dims(arr, 0)
        # Ajoute une dimension batch :
        # une image seule devient un batch de taille 1.

    def compute_query_embedding(self, image_path: str | Path) -> np.ndarray:
        """
        Calcule l'embedding de l'image requête.

        Le résultat est une représentation vectorielle de l'image
        dans l'espace latent appris par l'encodeur.
        """
        img = self._load_image(image_path)
        # Prépare l'image.

        emb = self.encoder(img, training=False).numpy()
        # Passe l'image dans l'encodeur en mode inférence
        # puis convertit la sortie TensorFlow en tableau NumPy.

        return emb
        # On conserve ici la sortie telle quelle, typiquement de forme `(1, D)`,
        # car `cosine_similarity` attend bien une matrice de requêtes.

    def top_k_similar(self, image_path: str | Path, k: int = 3):
        """
        Retourne les `k` images de la galerie les plus similaires à la requête.

        La similarité utilisée est la similarité cosinus entre :
        - l'embedding de la requête,
        - les embeddings pré-calculés de la galerie.

        Cette méthode exclut explicitement l'image requête elle-même
        afin d'éviter qu'elle ressorte en `Top 1`.
        """
        query_path = Path(image_path).resolve()
        # Chemin absolu de l'image requête.
        # On le garde pour les comparaisons de chemin avec les candidats.

        query_emb = self.compute_query_embedding(image_path)
        # Calcule l'embedding de l'image uploadée.

        sims = cosine_similarity(query_emb, self.embeddings)[0]
        # Compare l'embedding de requête à toute la galerie.
        # Résultat : un score de similarité par image de la base.
        # Le `[0]` extrait la première (et unique) ligne car on n'a qu'une requête.

        idxs = np.argsort(sims)[::-1]
        # Trie les indices des candidats par similarité décroissante.
        # On obtient donc d'abord les meilleurs voisins potentiels.

        query_hash = self._sha1_of_file(query_path)
        # Empreinte du contenu de la requête.
        # Elle sert à détecter une copie binaire exacte d'une image de la galerie.

        out = []
        # Liste finale des résultats qui sera retournée à l'interface.

        for i in idxs:
            candidate_path = Path(str(self.resolved_filenames[i])).resolve()
            # Récupère le chemin réel du candidat courant dans la galerie.

            # Exclut l'auto-match :
            # - même chemin de fichier,
            # - ou même contenu binaire si l'image a été uploadée via une copie/tempfile.
            if candidate_path == query_path:
                continue
                # Cas simple : la requête pointe exactement vers le même fichier.

            try:
                if self._sha1_of_file(candidate_path) == query_hash:
                    continue
                    # Cas plus subtil : la requête a été copiée dans un fichier temporaire
                    # ou sous un autre nom, mais son contenu est exactement le même.
            except OSError:
                # Si un fichier de galerie est illisible, on ne bloque pas toute la recherche.
                pass
                # C'est un choix de robustesse :
                # mieux vaut continuer la recherche que faire échouer toute l'interface.

            out.append({
                "filepath": str(candidate_path),
                # Chemin exploitable de l'image candidate.

                "similarity": float(sims[i]),
                # Score cosinus converti en float Python standard
                # pour faciliter l'affichage côté Streamlit.

                "label_idx": int(self.labels[i]),
                # Identifiant numérique de la classe.

                "style": str(self.classnames[int(self.labels[i])]),
                # Nom lisible du style correspondant au label.
            })
            if len(out) >= k:
                break
                # Dès qu'on a accumulé `k` vrais résultats, on peut s'arrêter.
                # C'est important car on parcourt potentiellement tous les candidats
                # jusqu'à compenser les exclusions d'auto-match.

        return out
        # Retourne une liste de dictionnaires prête à être consommée par l'app Streamlit.

    def explain_similarity(self, query_path: str | Path, candidate_path: str | Path) -> dict:
        """
        Produit une explication visuelle de similarité entre deux images via Grad-CAM.

        Cette méthode n'est utilisée que pour le meilleur résultat (`top-1`)
        quand l'utilisateur active l'option correspondante dans l'interface.
        """
        if self.gradcam is None:
            # Lazy loading : on ne construit l'objet Grad-CAM que lorsqu'il est demandé.
            try:
                from .gradcam_similarity import GradCamSimilarity
                self.gradcam = GradCamSimilarity(self.encoder, img_size=self.img_size)
                # On réutilise le même encodeur que pour le retrieval,
                # afin que l'explication soit cohérente avec le modèle réellement utilisé.
            except Exception as exc:
                self._gradcam_init_error = exc
                raise RuntimeError(
                    f"Grad-CAM indisponible: {exc}. Le top-k reste utilisable."
                ) from exc
                # Même si Grad-CAM n'est pas disponible, le retrieval standard reste fonctionnel.

        return self.gradcam.explain_similarity(query_path, candidate_path)
        # Délègue le calcul à l'objet spécialisé.
