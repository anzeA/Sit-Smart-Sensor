# setup.py file
from setuptools import setup, find_packages

setup(
    name="sit_smart_sensor",
    version="0.1",
    packages=find_packages(),
    package_dir={
        '': 'src',
    },
    requires=[
        "torch",
        "torchvision",
        "lightning"
        "opencv-python",
        "hydra-core",
        "numpy",
        "tensorboardx",
        "torchmetrics",
        "ax-platform",
        "playsound == 1.2.2",
        "grad-cam"

    ]
)
