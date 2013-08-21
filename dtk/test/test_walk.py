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

        self.time = np.linspace(0.0, 0.2, num=3)

        self.cycle = pandas.DataFrame(
            {
             'knee angle': np.sin(self.time),
             'ankle angle': np.cos(self.time),
             'knee moment': 5.0 + np.sin(self.time),
             'ankle moment': 5.0 + np.sin(self.time),
            }, index=self.time)

        self.all_cycles = pandas.Panel({'cycle1': self.cycle,
                                        'cycle2': self.cycle,
                                        'cycle3': self.cycle,
                                        'cycle4': self.cycle,
                                        })

        self.n = len(self.time)
        self.m = self.all_cycles.shape[0]
        self.p = len(self.sensors)
        self.q = len(self.controls)

        self.solver = SimpleControlSolver(self.all_cycles)
        self.solver.solve(self.sensors, self.controls)
        self.solver.lengths()

    def test_init(self):

        assert self.all_cycles is self.solver.data_panel
        assert self.sensors == self.solver.sensors
        assert self.controls == self.solver.controls

    def test_lengths(self):

        n, m, p, q = self.solver.lengths()

        assert self.solver.n == len(self.time)
        assert self.solver.m == self.all_cycles.shape[0]
        assert self.solver.p == len(self.sensors)
        assert self.solver.q == len(self.controls)

        assert n == len(self.time)
        assert m == self.all_cycles.shape[0]
        assert p == len(self.sensors)
        assert q == len(self.controls)

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

        #expected_b = np.array([])
        #for cycle in self.all_cycles.keys():
            #for t in self.time:
                #controls_at_t = self.all_cycles[cycle][self.controls].ix[t]
                #expected_b = np.hstack((expected_b, controls_at_t))

        expected_b = np.zeros(len(self.time) * len(self.controls))
        for i, t in enumerate(self.time):
            expected_b[i * len(self.controls):i * len(self.controls) + 2] = \
                -self.cycle[['knee moment',
                             'ankle moment']].ix[t]
        expected_b = np.hstack([expected_b for i in range(self.m)])

        A_cycle1 = np.zeros((self.n * self.q, self.n * self.q * (self.p + 1)))

        for i in range(self.n):
            A_cycle1[i * 2:i * 2 + 2, i * 6:i * 6 + 6] = \
                np.array(
                    [[self.cycle['knee angle'][i], self.cycle['ankle angle'][i], 0.0, 0.0, -1.0, 0.0],
                     [0.0, 0.0, self.cycle['knee angle'][i], self.cycle['ankle angle'][i], 0.0, -1.0]])

        # the cycles just happen to be the same
        cycles = [A_cycle1 for i in range(self.m)]
        expected_A = np.vstack(cycles)

        A, b = self.solver.form_a_b()

        testing.assert_allclose(expected_b, b)
        testing.assert_allclose(expected_A, A)

#TODO: Write a test for the gfr function.
