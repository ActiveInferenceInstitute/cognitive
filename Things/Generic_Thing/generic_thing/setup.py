from setuptools import setup, find_packages

setup(
    name="generic_thing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "networkx",
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
    ],
    python_requires=">=3.8",
    author="Active Inference Institute",
    description="A universal building block for Active Inference based systems",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 