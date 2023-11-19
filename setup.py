# setup.py file
from setuptools import setup, find_packages

from pathlib import Path
dir = Path(__file__).parent.absolute()
with open(dir / 'requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="sit_smart_sensor",
    version="0.2",
    packages=find_packages(),
    package_dir={
        '': 'src',
    },
    install_requires=required
)
