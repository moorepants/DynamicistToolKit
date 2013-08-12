#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from dtk.walk import find_constant_speed


def test_find_constant_speed():

    speed_array = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                          'data/treadmill-speed.csv'),
                             delimiter=',')
    time = speed_array[:, 0]
    speed = speed_array[:, 1]

    indice, constant_speed_time = find_constant_speed(time, speed, True)

    print constant_speed_time

    assert 6.5 < constant_speed_time < 7.5

#TODO test gfr
