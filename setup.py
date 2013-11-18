#!/usr/bin/env python

from setuptools import setup, find_packages

from dtk import __version__

setup(name='DynamicistToolKit',
      author='Jason K. Moore',
      author_email='moorepants@gmail.com',
      version=__version__,
      url="http://github.com/moorepants/DynamicistToolKit",
      description='Various tools for theoretical and experimental dynamics.',
      license='UNLICENSE.txt',
      packages=find_packages(),
      install_requires=['numpy>=1.7.0', 'scipy>=0.12.0', 'matplotlib>=1.2.0',
                        'pandas>=0.12.0'],
      extras_require={'doc': ['sphinx>=1.1.0', 'numpydoc>=0.4']},
      tests_require=['nose>1.3.0'],
      test_suite='nose.collector',
      long_description=open('README.rst').read(),
      classifiers=[
                   'Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Physics',
                  ],
      )
