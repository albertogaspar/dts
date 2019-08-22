"""
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='dts',
    version='0.0.0',
    author='Alberto Gasparin',
    install_requires=['keras', 'numpy', 'pandas', 'joblib', 'matplotlib',
                      'tqdm', 'scipy', 'requests', 'sacred', 'pymongo'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
)