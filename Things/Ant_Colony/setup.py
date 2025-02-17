from setuptools import setup, find_packages

setup(
    name="ant_colony",
    version="1.0.0",
    description="Multi-agent ant colony simulation",
    author="Your Name",
    packages=find_packages(),
    package_data={
        'ant_colony': ['config/*.yaml']
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'networkx',
        'pyyaml',
        'noise'  # For terrain generation
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'ant-colony=ant_colony.main:main',
        ],
    },
) 