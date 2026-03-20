from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "art-xplain"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.utils import PROJECT_ROOT, load_config, resolve_project_path


class RelativePathResolutionTests(unittest.TestCase):
    def test_default_config_loads_from_repo_root_cwd(self):
        previous_cwd = Path.cwd()
        try:
            os.chdir(REPO_ROOT)
            cfg = load_config()
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(cfg["project"]["name"], "Art-Xplain")

    def test_project_paths_resolve_from_app_root(self):
        cfg = load_config()

        self.assertEqual(PROJECT_ROOT, APP_ROOT.resolve())
        self.assertEqual(
            resolve_project_path(cfg["paths"]["embeddings_root"]),
            (APP_ROOT / "embeddings").resolve(),
        )
        self.assertEqual(
            resolve_project_path(cfg["paths"]["models_root"]),
            (APP_ROOT / "models").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
