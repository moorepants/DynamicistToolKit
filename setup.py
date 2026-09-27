#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('dtk/version.py').read())

setup(name='DynamicistToolKit',
      author='Jason K. Moore',
      author_email='moorepants@gmail.com',
      version=__version__,
      url="http://github.com/moorepants/DynamicistToolKit",
      description='Various tools for theoretical and experimental dynamics.',
      license='UNLICENSE.txt',
      packages=find_packages(),
      # Minimum dependency versions set to match Ubuntu 24.04 packages.
      install_requires=[
          'matplotlib>=3.6.3',
          'numpy>=1.26.4',
          'scipy>=1.11.4',
      ],
      extras_require={
          'doc': [
              'sphinx>=7.2.6',
              'numpydoc>=1.6.0',
          ],
      },
      tests_require=['pytest>=7.4.4'],
      long_description=open('README.rst').read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Programming Language :: Python :: 3.14',
          'Topic :: Scientific/Engineering :: Physics',
      ])
