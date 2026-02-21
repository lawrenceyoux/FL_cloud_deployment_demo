from setuptools import setup, find_packages

setup(
    name="fl-stroke-prediction",
    version="0.1.0",
    description="Federated Learning for multi-hospital stroke prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
