#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import os

# external
import numpy as np
from numpy import testing
import pandas
from nose.tools import assert_raises

# local
from ..walk import find_constant_speed, SimpleControlSolver, WalkingData
from ..process import time_vector


def test_find_constant_speed():

    speed_array = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                          'data/treadmill-speed.csv'),
                             delimiter=',')
    time = speed_array[:, 0]
    speed = speed_array[:, 1]

    indice, constant_speed_time = find_constant_speed(time, speed, plot=False)

    assert 6.5 < constant_speed_time < 7.5


class TestWalkingData():

    def setup(self):

        time = time_vector(1000, 100)

        omega = 2 * np.pi

        right_grf = 1000 * (0.75 + np.sin(omega * time))
        right_grf[right_grf < 0.0] = 0.0
        right_grf += 2.0 * np.random.normal(size=right_grf.shape)

        left_grf = 1000 * (0.75 + np.cos(omega * time))
        left_grf[left_grf < 0.0] = 0.0
        left_grf += 2.0 * np.random.normal(size=left_grf.shape)

        right_knee_angle = np.arange(len(time))
        right_knee_moment = np.arange(len(time))

        self.data_frame = \
            pandas.DataFrame({'Right Vertical GRF': right_grf,
                              'Left Vertical GRF': left_grf,
                              'Right Knee Angle': right_knee_angle,
                              'Right Knee Moment': right_knee_moment},
                             index=time)

        self.threshold = 10.0

    def test_init(self):

        walking_data = WalkingData(self.data_frame)

        assert walking_data.raw_data is self.data_frame

    def test_grf_landmarks(self, plot=False):

        walking_data = WalkingData(self.data_frame)

        right_strikes, left_strikes, right_offs, left_offs = \
            walking_data.grf_landmarks('Right Vertical GRF',
                                       'Left Vertical GRF',
                                       threshold=self.threshold,
                                       do_plot=plot)

        right_zero = self.data_frame['Right Vertical GRF'] < self.threshold
        instances = right_zero.apply(lambda x: 1 if x else 0).diff()
        expected_right_offs = \
            instances[instances == 1].index.values.astype(float)
        expected_right_strikes = \
            instances[instances == -1].index.values.astype(float)

        left_zero = self.data_frame['Left Vertical GRF'] < self.threshold
        instances = left_zero.apply(lambda x: 1 if x else 0).diff()
        expected_left_offs = \
            instances[instances == 1].index.values.astype(float)
        expected_left_strikes = \
            instances[instances == -1].index.values.astype(float)

        testing.assert_allclose(expected_right_offs, right_offs)
        testing.assert_allclose(expected_right_strikes, right_strikes)

        testing.assert_allclose(expected_left_offs, left_offs)
        testing.assert_allclose(expected_left_strikes, left_strikes)

    def test_interpolate(self):
        df = pandas.DataFrame({'a': [np.nan, 3.0, 5.0, 7.0],
                               'b': [5.0, np.nan, 9.0, 11.0],
                               'c': [2.0, 4.0, 6.0, 8.0],
                               'd': [0.5, 1.0, 1.5, np.nan]},
                              index=[0.0, 2.0, 4.0, 6.0])

        time = [0.0, 1.0, 3.0, 5.0]

        interpolated = WalkingData.interpolate(df, time)

        # NOTE : pandas.Series.interpolate does not extrapolate (because
        # np.interp doesn't.

        df_expected = pandas.DataFrame({'a': [4.0, 4.0, 4.0, 6.0],
                                        'b': [5.0, 6.0, 8.0, 10.0],
                                        'c': [2.0, 3.0, 5.0, 7.0],
                                        'd': [0.5, 0.75, 1.25, 1.5]},
                                       index=time)

        testing.assert_allclose(interpolated.values, df_expected.values)

        testing.assert_allclose(interpolated.values, df_expected.values)
        testing.assert_allclose(interpolated.index.values.astype(float),
                                df_expected.index.values.astype(float))

    def test_split_at(self, plot=False):

        walking_data = WalkingData(self.data_frame)
        walking_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)

        side = 'right'
        series = 'Right Vertical GRF'

        steps = walking_data.split_at('right')

        for i, step in steps.iteritems():
            start_step = walking_data.strikes[side][i]
            end_step = walking_data.strikes[side][i + 1]
            testing.assert_allclose(step[series],
                walking_data.raw_data[series][start_step:end_step])

        if plot is True:
            walking_data.plot_steps(series, 'Left Vertical GRF')

        steps = walking_data.split_at(side, 'stance')

        for i, step in steps.iteritems():
            start_step = walking_data.strikes[side][i]
            end_step = walking_data.offs[side][i + 1]
            testing.assert_allclose(step[series],
                walking_data.raw_data[series][start_step:end_step])

        if plot is True:
            walking_data.plot_steps(series, 'Left Vertical GRF')

        steps = walking_data.split_at(side, 'swing')

        for i, step in steps.iteritems():
            start_step = walking_data.offs[side][i]
            end_step = walking_data.strikes[side][i]
            testing.assert_allclose(step[series],
                walking_data.raw_data[series][start_step:end_step])

        if plot is True:
            walking_data.plot_steps(series, 'Left Vertical GRF')
            import matplotlib.pyplot as plt
            plt.show()

    def test_plot_steps(self):

        walking_data = WalkingData(self.data_frame)
        walking_data.grf_landmarks('Right Vertical GRF',
                                   'Left Vertical GRF',
                                   threshold=self.threshold)
        walking_data.split_at('right')

        assert_raises(ValueError, walking_data.plot_steps)


class TestSimpleControlSolver():

    def setup(self):

        self.sensors = ['knee angle', 'ankle angle']
        self.controls = ['knee moment', 'ankle moment']

        self.time = np.linspace(0.0, 5.0, num=100)

        self.n = len(self.time)
        self.p = len(self.sensors)
        self.q = len(self.controls)
        self.r = 10
        self.m = self.r / 2

        self.gain_omission_matrix = np.array(self.q * [self.p * [True]])
        self.gain_omission_matrix[0, 1] = False
        self.gain_omission_matrix[1, 0] = False

        # pick m*(t), K(t), and s(t), then generate mm(t)
        # mm(t) = m*(t) - K(t) * s(t)

        # TODO : maybe I should set the problem up more generally, like:
        # mm(t) = m0(t) + K(t) * [s0(t) - s(t)]

        self.m_star = 100.0 * np.array([np.cos(self.time),
                                        np.sin(self.time)]).T
        self.K = np.array([[np.sin(self.time), np.cos(self.time)],
                           [np.sin(2.0 * self.time),
                            np.cos(3.0 * self.time)]]).T

        self.s = np.zeros((self.r, self.n, self.p))
        self.mm = np.zeros((self.r, self.n, self.q))
        cycles = {}
        for i in range(self.r):
            noise = 0.25 * np.random.randn(self.n, self.p)
            self.s[i] = np.vstack(((i + 1) * np.sin(self.time),
                                   (i + 1) * np.cos(self.time))).T + noise
            for j in range(self.n):
                self.mm[i, j] = self.m_star[j] - np.dot(self.K[j],
                                                        self.s[i, j])
            cycles['cycle {}'.format(i)] = \
                pandas.DataFrame({self.sensors[0]: self.s[i, :, 0],
                                  self.sensors[1]: self.s[i, :, 1],
                                  self.controls[0]: self.mm[i, :, 0],
                                  self.controls[1]: self.mm[i, :, 1],
                                  }, index=self.time)

        self.all_cycles = pandas.Panel(cycles)

        self.solver = SimpleControlSolver(self.all_cycles, self.sensors,
                                          self.controls)

    def test_init(self):

        assert self.all_cycles.iloc[:self.m] == self.solver.identification_data
        assert self.all_cycles.iloc[self.m:] == self.solver.validation_data
        assert self.solver.n == self.n
        assert self.solver.m == self.m

        assert self.sensors is self.solver.sensors
        assert self.solver.p == self.p

        assert self.controls is self.solver.controls
        assert self.solver.q == self.q

        self.solver = SimpleControlSolver(self.all_cycles, self.sensors,
                                          self.controls,
                                          validation_data=self.all_cycles)

        assert self.all_cycles is self.solver.identification_data
        assert self.all_cycles is self.solver.validation_data
        assert self.solver.n == self.n
        assert self.solver.m == self.r

    def test_form_sensor_vectors(self):

        sensor_vectors = self.solver.form_sensor_vectors()

        # this should be an m x n x p array
        assert sensor_vectors.shape == (self.m, self.n, self.p)

        for i in range(self.m):
            for j in range(self.n):
                testing.assert_allclose(sensor_vectors[i, j],
                    self.all_cycles.iloc[i][self.sensors].iloc[j])

    def test_form_control_vectors(self):

        control_vectors = self.solver.form_control_vectors()

        # this should be an m x n x q array
        assert control_vectors.shape == (self.m, self.n, self.q)

        for i in range(self.m):
            for j in range(self.n):
                testing.assert_allclose(control_vectors[i, j],
                    self.all_cycles.iloc[i][self.controls].iloc[j])

    def test_form_a_b(self):

        for cycle_key in sorted(self.all_cycles.keys())[:self.m]:

            cycle = self.all_cycles[cycle_key]

            A_cycle = np.zeros((self.n * self.q,
                                self.n * self.q * (self.p + 1)))

            for i, t in enumerate(self.time):
                controls_at_t = cycle[self.controls].ix[t]
                try:
                    expected_b = np.hstack((expected_b, controls_at_t))
                except NameError:
                    expected_b = controls_at_t

                if self.p > 2:
                    raise Exception("This test only works for having two sensors.")

                A_cycle[i * 2:i * 2 + 2, i * 6:i * 6 + 6] = \
                    np.array(
                        [[-cycle[self.sensors[0]][t], -cycle[self.sensors[1]][t], 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, -cycle[self.sensors[0]][t], -cycle[self.sensors[1]][t], 0.0, 1.0]])
            try:
                expected_A = np.vstack((expected_A, A_cycle))
            except NameError:
                expected_A = A_cycle

        A, b = self.solver.form_a_b()

        testing.assert_allclose(expected_b, b)
        testing.assert_allclose(expected_A, A)

        # Now test to see if the gain omission works.
        self.solver.gain_omission_matrix = self.gain_omission_matrix

        A, b = self.solver.form_a_b()

        # TODO : This is the extact code in the source, not sure it is a
        # useful test then... It would be more useful if it was a different
        # implmentation or something.
        # form a x vector from the gain_omission_matrix
        x1 = self.gain_omission_matrix.reshape(self.q * self.p)
        x2 = np.array(self.q * [True])
        for i in range(self.n):
            try:
                x_total = np.hstack((x_total, x1, x2))
            except NameError:
                x_total = np.hstack((x1, x2))
        # x has nans that correspond to the columns in A
        expected_A = expected_A[:, x_total]

        testing.assert_allclose(expected_A, A)

    def test_least_squares(self):

        A, b = self.solver.form_a_b()
        x, variance, covariance = self.solver.least_squares(A, b)
        for i in range(self.n):
            try:
                expected_x = np.hstack((expected_x, self.K[i].flatten(),
                                        self.m_star[i]))
            except NameError:
                expected_x = np.hstack((self.K[i].flatten(), self.m_star[i]))
        testing.assert_allclose(x, expected_x, atol=1e-10)

        self.solver.gain_omission_matrix = self.gain_omission_matrix
        A, b = self.solver.form_a_b()
        x, variance, covariance = self.solver.least_squares(A, b)

        expected_normal_x_length = self.q * (1 + self.p) * self.n
        removed_parameters = self.n * self.gain_omission_matrix.sum()

        assert len(x) == expected_normal_x_length - removed_parameters
        # TODO : check that x is correct

    def test_deconstruct_solution(self):
        # TODO : I don't know what the variances should be for this problem,
        # so I don't have a check for that in place yet.

        A, b = self.solver.form_a_b()
        x, variance, covariance = self.solver.least_squares(A, b)
        (gain_matrices, control_vectors, gain_matrices_variance,
         control_vectors_variance) = \
            self.solver.deconstruct_solution(x, covariance)

        testing.assert_allclose(gain_matrices, self.K, atol=1e-10)
        testing.assert_allclose(control_vectors, self.m_star, atol=1e-12)

        # now with gain omission matrix
        self.solver.gain_omission_matrix = self.gain_omission_matrix
        A, b = self.solver.form_a_b()
        x, variance, covariance = self.solver.least_squares(A, b)
        (gain_matrices, control_vectors, gain_matrices_variance,
         control_vectors_variance) = \
            self.solver.deconstruct_solution(x, covariance)

        for i in range(self.n):
            testing.assert_equal(np.abs(gain_matrices[i]) > 1e-16,
                                 self.gain_omission_matrix)
            testing.assert_equal(np.abs(gain_matrices_variance[i]) > 1e-16,
                                 self.gain_omission_matrix)

    def test_solve(self):

        self.solver.solve()
        assert self.solver.gain_omission_matrix is None

        self.solver.solve(gain_omission_matrix=self.gain_omission_matrix)

        testing.assert_equal(self.solver.gain_omission_matrix,
                             self.gain_omission_matrix)

        # TODO : check everything else in solve!

    def test_compute_estimated_controls(self):
        A, b = self.solver.form_a_b()
        x, variance, covariance = self.solver.least_squares(A, b)
        solution = self.solver.deconstruct_solution(x, covariance)

        estimated = self.solver.compute_estimated_controls(solution[0],
                                                           solution[1])

        # mean across validation steps, n x p
        expected_s0 = self.s[self.m:].mean(axis=0)

        # for each step
        for i, (item, df) in enumerate(estimated.iteritems()):
            # check if the measured moments match
            testing.assert_allclose(df[self.controls].values, self.mm[i + self.m])
            # check if m* matches
            testing.assert_allclose(df[[c + '*' for c in
                                        self.controls]].values, self.m_star,
                                    atol=1e-10)

            for j in range(self.n):
                expected_m0 = self.mm[i + self.m, j] - \
                    np.dot(self.K[j], (expected_s0[j] - self.s[i + self.m, j]))
                m0 = df.loc[df.index[j], [c + '0' for c in self.controls]].values
                testing.assert_allclose(m0, expected_m0)

            # todo: check the control contributions from the error in the
            # signals

#TODO: Write a test for the gfr function.
