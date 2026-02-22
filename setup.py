from setuptools import setup, find_packages

setup(
    name="fl-stroke-prediction",
    version="0.1.0",
    description="Federated Learning for multi-hospital stroke prediction",
    # find_packages() discovers src, src.models, src.federated, src.preprocessing,
    # src.utils — installed with the 'src.*' prefix matching all import statements.
    packages=find_packages(),
    python_requires=">=3.10",
)
