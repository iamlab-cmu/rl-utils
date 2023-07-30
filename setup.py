"""
rl-utils
"""

import io
import os
import re
from setuptools import setup

requirements = [
    'numpy',
    'pytest',
    'scipy'
]

def package_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file_path = os.path.join(current_dir, "rl_utils", "__init__.py")
    with io.open(version_file_path, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(name='rl-utils',
      version=package_version(),
      description='A library for reinforcement learning algorithms, including REPS.',
      author='Tabitha Lee',
      author_email='tabithalee@cmu.edu',
      packages=['rl_utils'],
      package_dir={'': '.'},
      install_requires = requirements
      )
