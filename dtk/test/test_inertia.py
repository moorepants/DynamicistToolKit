#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external libraries
import numpy as np
from numpy import testing

# local libraries
from ..inertia import *


def test_euler_rotation(display=False):
    # body-three 1-2-3
    a = [15., 0., 0.]
    order = (1, 2, 3)
    R = euler_rotation(a, order)
    C = np.array([[1., 0., 0.],
                  [0., np.cos(a[0]), np.sin(a[0])],
                  [0., -np.sin(a[0]), np.cos(a[0])]])
    if display:
        print("body-three 1-2-3")
        print(R)
        print(C)
        print('-' * 79)

    testing.assert_almost_equal(R, C)

    # body-three 1-2-3
    a = np.array([0.34, 23.6, -0.2])
    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    c3 = np.cos(a[2])
    s1 = np.sin(a[0])
    s2 = np.sin(a[1])
    s3 = np.sin(a[2])

    order = (1, 2, 3)
    R = euler_rotation(a, order)

    # definition of body 1-2-3 rotations from Spacecraft Dynamics, Kane,
    # Likins, Levinson, 1982
    C = np.array([[c2 * c3, s1 * s2 * c3 + s3 * c1, -c1 * s2 * c3 + s3 * s1],
                   [-c2 * s3, -s1 * s2 * s3 + c3 * c1, c1 * s2 * s3 + c3 *s1],
                   [s2, -s1 * c2, c1 * c2]])

    if display:
        print("body-three 1-2-3")
        print(R)
        print(C)
        print('-' * 79)

    testing.assert_almost_equal(R, C)

    # test 3-1-3
    a = np.array([1., 1.1, 1.2])
    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    c3 = np.cos(a[2])
    s1 = np.sin(a[0])
    s2 = np.sin(a[1])
    s3 = np.sin(a[2])

    order = (3, 1, 3)

    R = euler_rotation(a, order)

    # definition of body 3-1-3 rotations from Spacecraft Dynamics, Kane,
    # Likins, Levinson, 1982
    C = np.array([[-s1 * c2 * s3 + c3 * c1, c1 * c2 * s3 + c3 *s1, s2 *s3],
                  [-s1 * c2 * c3 - s3 * c1, c1 * c2 * c3 - s3 * s1, s2 * c3],
                  [s1 * s2, -c1 * s2, c2]])
    if display:
        print("body-two 3-1-")
        print(R)
        print(C)
        print('-' * 79)

    testing.assert_almost_equal(R, C)

    # test 1-3-2
    a = np.array([0.234, 0.0004, 0.50505])
    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    c3 = np.cos(a[2])
    s1 = np.sin(a[0])
    s2 = np.sin(a[1])
    s3 = np.sin(a[2])

    order = (1, 3, 2)

    R = euler_rotation(a, order)

    # definition of body-three 1-3-2 rotations from Spacecraft Dynamics, Kane,
    # Likins, Levinson, 1982
    C = np.array([[c2 * c3, c1 * s2 * c3 + s3 * s1, s1 * s2 * c3 - s3 * c1],
                   [-s2, c1 * c2, s1 * c2],
                   [c2 * s3, c1 * s2 * s3 - c3 * s1, s1 * s2 * s3 + c3 * c1]])

    if display:
        print('-' * 79)
        print("body-three 1-3-2")
        print(R)
        print(C)

    testing.assert_almost_equal(R, C)

    # test 2-1-3
    a = np.array([0.234, 0.0004, 0.50505])
    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    c3 = np.cos(a[2])
    s1 = np.sin(a[0])
    s2 = np.sin(a[1])
    s3 = np.sin(a[2])

    order = (2, 1, 3)

    R = euler_rotation(a, order)

    # definition of body 2-1-3 rotations from Spacecraft Dynamics, Kane,
    # Likins, Levinson, 1982
    C = np.array([[s1 * s2 * s3 + c3 * c1, c2 * s3, c1 * s2 * s3 - c3 * s1],
                   [s1 * s2 * c3 - s3 * c1, c2 * c3, c1 * s2 * c3 + s3 * s1],
                   [s1 * c2, -s2, c1 * c2]])

    if display:
        print('-' * 79)
        print("body-three 2-1-3")
        print(R)
        print(C)
    testing.assert_almost_equal(R, C)
