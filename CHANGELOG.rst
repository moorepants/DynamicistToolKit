=============
Release Notes
=============

0.5.3
=====

- Added the license and readme to the source distriubtion.

0.5.2
=====

- Screwed up pypi upload on 0.5.1, so bumping one more time.

0.5.1
=====

- Import nanmean from numpy instead of scipy and fix float slices. [PR `#34`_]

.. _#34: https://github.com/moorepants/DynamicistToolKit/pull/34

0.5.0
=====

- bicycle.py functions now output numpy arrays instead of matrices.
- Support for Python 3 [PR `#30`_ and `#32`_].

.. _#30: https://github.com/moorepants/DynamicistToolKit/pull/30
.. _#32: https://github.com/moorepants/DynamicistToolKit/pull/32

0.4.0
=====

- Made the numerical derivative function more robust and featureful. [PR
  `#27`_]
- ``butterworth`` now uses a corrected cutoff frequency to adjust for the
  double filtering. [PR `#28`_]

.. _#27: https://github.com/moorepants/DynamicistToolKit/pull/27
.. _#28: https://github.com/moorepants/DynamicistToolKit/pull/28

0.3.5
=====

- Fixed bug in coefficient_of_determination. [PR `#23`_]

.. _#23: https://github.com/moorepants/DynamicistToolKit/pull/23

0.3.4
=====

- Fixed bug in normalized cutoff frequency calculation. [PR `#21`_]

.. _#21: https://github.com/moorepants/DynamicistToolKit/pull/21

0.3.2
=====

- Fixed bug in butterworth function and added tests.

0.3.1
=====

- Fixed butterworth to work with SciPy 0.9.0. [PR `#18`_]

.. _#18: https://github.com/moorepants/DynamicistToolKit/pull/18

0.3.0
=====

- Removed pandas dependency.
- Improved time vector function.
- Removed gait analysis code (walk.py), now at
  http://github.com/csu-hmc/Gait-Analysis-Toolkit.
- TravisCI tests now run, added image to readme.
- Added documentation at ReadTheDocs.

0.2.0
=====

- Addition of walking dynamics module.

0.1.0
=====

- Original code base that was used for the computations in this dissertation:
  https://github.com/moorepants/dissertation
