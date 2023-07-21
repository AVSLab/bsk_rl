from setuptools import setup

setup(
    name="bsk_rl",
    version="0.0.0",
    install_requires=[
        "gymnasium",
        "matplotlib",
        "numpy",
        "pandas",
        "stable-baselines3",
        "torch",
        "tensorflow",
        "deap==1.3.3",
        "scikit-learn",
    ],
)
