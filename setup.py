#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='fraud-detection',
      version='1.0',
      description='Credit Cards Fraud Detection',
      author='Yanina Kutovaya',
      author_email='kutovaiayp@yandex.ru',
      url='https://github.com/Yanina-Kutovaya/fraud-detection',
      package_dir={"": "src"},
      packages=find_packages(where="src"),
     )