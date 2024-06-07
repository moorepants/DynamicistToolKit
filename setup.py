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
      # Minimum dependency versions set to match Ubuntu 22.04 packages.
      install_requires=[
          'matplotlib>=3.5.1',
          'numpy>=1.21.5',
          'scipy>=1.8.0',
      ],
      extras_require={
          'doc': [
              'sphinx>=4.3.2',
              'numpydoc>=1.2',
          ],
      },
      tests_require=['nose>1.3.0'],
      test_suite='nose.collector',
      long_description=open('README.rst').read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics',
      ])
