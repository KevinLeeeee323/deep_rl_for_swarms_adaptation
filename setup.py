from setuptools import find_packages, setup

setup(
    name="deep_rl_for_swarms",
    version="0.1.0",
    packages=find_packages(include=["deep_rl_for_swarms", "deep_rl_for_swarms.*"]),
    python_requires=">=3.10,<3.11",
    description="Continuous pursuit environment with a PyTorch MAPPO training pipeline",
    install_requires=[
        "numpy==1.26.4", "scipy==1.13.1", "networkx==2.8.8",
        "shapely==1.8.5.post1", "fast-histogram==0.14", "gym==0.23.1",
        "matplotlib==3.8.4", "torch==2.5.1",
        "imageio==2.37.0", "imageio-ffmpeg==0.6.0",
    ],
)
