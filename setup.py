from setuptools import find_packages
from setuptools import setup
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with (ROOT / "requirements.txt").open() as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='art-xplain',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/art-xplain-run'],
      zip_safe=False)
