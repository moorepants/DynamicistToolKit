from distutils.core import setup

setup(name='DynamicistToolKit',
      author='Jason Moore',
      author_email='moorepants@gmail.com',
      version='0.1.0dev',
      packages=['dtk'],
      install_requires=['numpy', 'scipy'],
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      long_description=open('README.txt').read(),
      )
