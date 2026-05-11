from setuptools import setup, find_packages

setup(
    name="linkbridge-finance",
    version="0.1.0",
    description="Financial indicator computation engine for LinkBridge",
    author="LinkBridge",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.2.0",
        "pandas>=2.2.0",
        "scipy>=1.14.0",
        "statsmodels>=0.14.0",
    ],
)
