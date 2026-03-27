from __future__ import annotations

import sys
from pathlib import Path


# Ce petit point d'entrée existe pour lancer Streamlit depuis `src/`
# tout en garantissant que la racine métier du projet est bien sur `sys.path`.
# Sans cela, l'exécution via `streamlit run art-xplain/src/app_streamlit.py`
# peut échouer sur les imports `from src...` selon le répertoire courant.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.front_end.app import run_app


# Toute la logique d'interface reste centralisée dans `front_end/app.py`.
# Le fichier courant doit rester volontairement minimal pour :
# - simplifier le démarrage local,
# - éviter de dupliquer de la logique de bootstrap,
# - conserver un point d'entrée stable pour l'équipe.
run_app()
