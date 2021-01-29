# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Deep Deterministic Policy Gradient'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@mail.mil'
__status__ = 'Dev'

setup(name='maddpg',
      version='0.0.1',
      description='Multi-Agent Deep Deterministic Policy Gradient',
      url='',
      author='Rolando Fernandez',
      author_email='rolando.fernandez1.civ@mail.mil',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'joblib', 'matplotlib', 'numpy', 'numpy-stl', 'skccm', 'tensorflow==1.15', 'pillow==7.2.0'])
