<h1 align="center">Art-Xplain</h2>
<h3 align="center">Moteur de similarité stylistique pour oeuvres peintes</h2>


---
- **Emmanuelle** - _manievfoulards@gmail.com_
- **Lucile** - _lucilejosse.mail@gmail.com_
- **Lionel** - _lion94.home@gmail.com_

Art-Xplain est un projet Python/TensorFlow qui apprend un encodeur visuel pour comparer des œuvres d'art par similarité de style.

Le pipeline couvre:
- préparation d'un dataset Keras-ready (train/val/test)
- entraînement d'un model encodeur
- recherche top-k par similarité cosinus
- explication visuelle de similarité avec Grad-CAM
- démo interactive Streamlit

## 1) Structure des données

Entrée (dataset Kaggle source):
- `data/in/kaggle-wikiart`

Sortie (dataset généré pour entraînement):
- `data/out/train/<style>/*.jpg`
- `data/out/val/<style>/*.jpg`
- `data/out/test/<style>/*.jpg`

Ces chemins sont configurés dans `config.yaml` via:
- `paths.kaggle_root: data/in/kaggle-wikiart`
- `paths.keras_root: data/out`

## 2) Installation

```bash
pyenv virtualenv art-xplain
cd art-xplain
pyenv local art-xplain
pip install -r requirements.txt
```

## 3) Pipeline complet

### Étape A — Construire les splits train/val/test

- _cf. notebook `step_1_build_dataset_step_by_step.ipynb`_

### Étape B — Entraîner le model encodeur

### Étape C — Calculer les embeddings

### Étape D — Projeter en 2D (UMAP)

### Étape E — Lancer l'application Streamlit

### Qu'est-ce que UMAP ?

UMAP (Uniform Manifold Approximation and Projection) est une méthode de réduction de dimension.

Dans ce projet, UMAP prend les embeddings haute dimension (`vectors.npy`) et les projette en 2D (`latent_2d.npy`) pour visualiser les œuvres sous forme de points.

Intuition:
- les œuvres proches dans l'espace d'embeddings restent proches sur la carte 2D
- les groupes/amas visibles correspondent souvent à des styles similaires

Paramètres principaux (`config.yaml`):
- `umap.n_neighbors`: taille du voisinage local utilisé par UMAP
- `umap.min_dist`: contrôle le degré de compacité des points en 2D


## 5) Notebook de préparation

Définition (materialization): dans ce projet, la materialization correspond à la copie physique des images vers l’arborescence cible `data/out/train|val|test/<style>/...` à partir des splits calculés.

Le notebook étape 1 permet de travailler, tester, comprendre et valider la préparation du dataset labelisé pour entrainer du model:
- `notebooks/step_1_build_dataset_step_by_step.ipynb`

Le notebook `step_1_build_dataset_step_by_step.ipynb` exécute les opérations lecture CSV, préparation labels, filtrage, split, nettoyage, matérialisation.

Les fonctions `detect_images_root_from_filenames`, `infer_label_from_filename_parent`, `normalize_label_value`, `clean_output_root` et `materialize_split` sont elles codées dans `src/build_dataset_from_csv.py`. Le notebook permet d’exécuter ces opérations de base pas à pas.

### Résumé des cellules du notebook

- Cellules 1-2: imports, détection de la racine projet, chargement de la config.
- Cellule 3: lecture du CSV et inspection des colonnes.
- Cellule 4: préparation de `filename` + `label` (inférence/normalisation).
- Cellule 5: détection du dossier images + filtrage des styles.
- Cellule 6: split stratifié `train/val/test`.
- Cellule 7: nettoyage optionnel de `data/out` (`clean_output_root`).
- Cellule 8: matérialisation optionnelle des splits (`materialize_split`).
- Cellule 9: vérification rapide du résultat (comptage styles/fichiers).

### Résumé des fonctions clés du notebook

- `detect_images_root_from_filenames`:
  teste plusieurs racines candidates et sélectionne celle qui résout le plus de chemins `filename` du CSV (ex: `kaggle_root`, `kaggle_root/images`, sous-dossiers).

- `infer_label_from_filename_parent`:
  essaie d'inférer le label depuis le dossier parent du `filename` (ex: `Impressionism/img.jpg` -> `Impressionism`), utile quand la colonne `style` n'est pas fiable ou absente.

- `normalize_label_value`:
  nettoie/normalise les labels (gestion des labels stockés comme listes texte, suppression d'ambiguïtés, remplacement de `/` par `_` pour créer des dossiers sûrs).

- `clean_output_root`:
  supprime le contenu de `paths.keras_root` (`data/out`) pour repartir d'un état propre avant une nouvelle génération.

- `materialize_split`:
  copie les images dans la structure finale `train/val/test/<style>/...` en résolvant les chemins sources et en comptant les fichiers copiés/manquants.


## 6) Dépendances principales

- TensorFlow
- NumPy
- Pandas
- scikit-learn
- UMAP
- OpenCV
- Streamlit

## 7) Model: (description du model choisi)

  _- A faire pour notre session du mardi 10 (en cours LN)_


## 8) Notes

- Le script de build dataset est tolérant aux variations de format CSV et peut inférer le label depuis le dossier parent de `filename`.
