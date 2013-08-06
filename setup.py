from setuptools import setup, find_packages

setup(name='DynamicistToolKit',
      author='Jason Moore',
      author_email='moorepants@gmail.com',
      version='0.1.0',
      url="http://github.com/moorepants/DynamicistToolKit",
      description='Various tool kits for theorectical and experimental dynamics.',
      license='UNLICENSE.txt',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib'],
      tests_require=['nose'],
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
