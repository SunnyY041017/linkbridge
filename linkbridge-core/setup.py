from setuptools import setup, find_packages

setup(
    name="linkbridge-core",
    version="0.1.0",
    description="Multi-Agent orchestration framework for LinkBridge",
    author="LinkBridge",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "openai>=1.58.0",
        "pyyaml>=6.0",
        "pydantic>=2.10.0",
    ],
)
