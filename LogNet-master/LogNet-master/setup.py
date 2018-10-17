'''Setup script for lognet'''

import os
from setuptools import setup, find_packages

__version__ = '0.0'

with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()


setup(name='lognet',
      version = __version__,
      description = 'Machine Learning with Log ASCII Standard (LAS) files',
      long_description = 'Machine Learning with Log ASCII Standard (LAS) files',
      packages=find_packages(exclude=['docs', 'tests']),
      install_requires=requirements
     )


