============
Contributing
============

The name of this distribution is "DynamicistToolKit" but it is best to think of
it as a shared collection of tested tools for the TU Delft Bicycle Lab. Our
primary audience is ourselves, but since this is public it may garner other
users if we make anything more broadly useful. So most anything useful to the
science and engineering we do can go in here. Secondly, think of this as a
first location to graduate useful tools you have written for your own research.

There are several guidelines we should follow to ensure we can use this as a
shared dependency in our work:

1. Do not change input/output behavior of any function, method, or class if it
   has been released (version pushed to PyPi & Conda Forge).
2. If you absolutely have to change input/ouput behavior then, first, consider
   making a new function with your desired behavior or, secondly, deprecate the
   function for at least 1 year and at least 1 version release with a
   deprecation warning.
3. If a function produces the incorrect (mathematical) result you may change
   the output without deprecation (in general).
4. Functions, methods, classes, modules that start with a single underscore
   ``_`` are considered private and you may change the input/output behavior.
   Everything else is considered public and the API is frozen when we release
   the software.
5. All additions must include docstrings with at least one example of use and
   enough unit tests to protect against regressions, i.e. ensure that the
   input/output API cannot be changed without tests failing.

If we follow these basic guidelines we should have a relatively stable tool
that we can all use and not worry about our code breaking when others do work
on it.
