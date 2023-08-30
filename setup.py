"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

__author__ = "Dan Von Pasecky"
__email__ = "danvonpasecky@gmail.com"
__version__ = "0.4.0"
__license__ = "MIT"

import pathlib

from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fuzzy-matcher",
    version=__version__,
    description="Levenshtein Distance Matcher",
    long_description=long_description,
    python_requires=">=3.9",
    license=__license__,
    author=__author__,
    author_email=__email__,
    url="https://github.com/dvonpasecky/fuzzy-matcher",
)
