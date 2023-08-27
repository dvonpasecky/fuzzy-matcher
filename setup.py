"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="sample",
    version="0.1.0",
    python_requires=">=3.11, >=3.10, >=3.9, >=3.8, >=3.7, >=3.6, >=3.5, >=2.7",
    description="Sample package",
    long_description=long_description,
    license="MIT",
    author="Dan Von Pasecky",
    author_email="danvonpasecky@gmail.com",
    url="https://github.com/dvonpasecky/template",
    packages=find_packages(exclude=("tests", "docs")),
)
