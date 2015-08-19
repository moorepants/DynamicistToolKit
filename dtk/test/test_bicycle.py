#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libary
from math import pi

# local libraries
from .. import bicycle


def test_basu_to_moore_input():

    basu = bicycle.basu_table_one_input()

    lam = pi / 10.
    rr = 0.3

    mooreInput = bicycle.basu_to_moore_input(basu, rr, lam)
    for k, v in mooreInput.items():
        print(k, ':', v)
