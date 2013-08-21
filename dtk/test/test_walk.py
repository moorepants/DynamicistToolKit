#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import os

# external
import numpy as np
from numpy import testing
import pandas

# local
from dtk.walk import find_constant_speed, SimpleControlSolver


def test_find_constant_speed():

    speed_array = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                          'data/treadmill-speed.csv'),
                             delimiter=',')
    time = speed_array[:, 0]
    speed = speed_array[:, 1]

    indice, constant_speed_time = find_constant_speed(time, speed, plot=False)

    assert 6.5 < constant_speed_time < 7.5


class TestSimpleControlSolver():

    def setup(self):

        self.sensors = ['knee angle', 'ankle angle']
        self.controls = ['knee moment', 'ankle moment']

        self.time = np.linspace(0.0, 5.0, num=100)

        self.n = len(self.time)
        self.p = len(self.sensors)
        self.q = len(self.controls)
        self.m = 4

        # pick m*, K and s, then generate m
        # m(t) = m*(t) + K(t)s(t)

        self.m_star = np.ones((self.n, self.q))
        self.K = np.array([[np.sin(self.time), np.cos(self.time)],
                           [self.time ** 2, self.time ** 3]]).T

        s = np.zeros((self.m, self.n, self.p))
        m = np.zeros((self.m, self.n, self.q))
        cycles = {}
        noise = 0.05 * np.random.randn(self.n, self.p)
        for i in range(self.m):
            s[i] = np.vstack(((i + 1) * np.sin(self.time),
                              (i + 1) * np.cos(self.time))).T + noise
            for j in range(self.n):
                m[i, j] = self.m_star[j] + np.dot(self.K[j], s[i, j])
            cycles['cycle {}'.format(i)] = \
                pandas.DataFrame({self.sensors[0]: s[i, :, 0],
                                  self.sensors[1]: s[i, :, 1],
                                  self.controls[0]: m[i, :, 0],
                                  self.controls[1]: m[i, :, 1],
                                  }, index=self.time)

        self.all_cycles = pandas.Panel(cycles)

        self.solver = SimpleControlSolver(self.all_cycles, self.sensors,
                self.controls)

    def test_init(self):

        assert self.all_cycles is self.solver.data_panel
        assert self.sensors == self.solver.sensors
        assert self.controls == self.solver.controls

    def test_lengths(self):

        n, m, p, q = self.solver.lengths()

        assert self.solver.n == self.n
        assert self.solver.m == self.m
        assert self.solver.p == self.p
        assert self.solver.q == self.q

        assert n == self.n
        assert m == self.m
        assert p == self.p
        assert q == self.q

    def test_form_sensor_vectors(self):

        sensor_vectors = self.solver.form_sensor_vectors()
        # this should be an m x n x p array
        assert sensor_vectors.shape == (self.m, self.n, self.p)

        for i in range(self.m):
            for j in range(self.n):
                testing.assert_allclose(sensor_vectors[i, j],
                    self.all_cycles.ix[i][self.sensors].ix[j])

    def test_form_control_vectors(self):

        control_vectors = self.solver.form_control_vectors()

        # this should be an m x n x q array
        assert control_vectors.shape == (self.m, self.n, self.q)

        for i in range(self.m):
            for j in range(self.n):
                testing.assert_allclose(control_vectors[i, j],
                    self.all_cycles.ix[i][self.controls].ix[j])

    def test_form_a_b(self):

        for cycle_key in sorted(self.all_cycles.keys()):

            cycle = self.all_cycles[cycle_key]

            A_cycle = np.zeros((self.n * self.q,
                                self.n * self.q * (self.p + 1)))

            for i, t in enumerate(self.time):
                controls_at_t = -cycle[self.controls].ix[t]
                try:
                    expected_b = np.hstack((expected_b, controls_at_t))
                except NameError:
                    expected_b = controls_at_t

                if self.p > 2:
                    raise Exception("This test only works for having two sensors.")

                A_cycle[i * 2:i * 2 + 2, i * 6:i * 6 + 6] = \
                    np.array(
                        [[cycle[self.sensors[0]][t], cycle[self.sensors[1]][t], 0.0, 0.0, -1.0, 0.0],
                         [0.0, 0.0, cycle[self.sensors[0]][t], cycle[self.sensors[1]][t], 0.0, -1.0]])
            try:
                expected_A = np.vstack((expected_A, A_cycle))
            except NameError:
                expected_A = A_cycle

        A, b = self.solver.form_a_b()

        testing.assert_allclose(expected_b, b)
        testing.assert_allclose(expected_A, A)

#TODO: Write a test for the gfr function.
