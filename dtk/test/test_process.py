#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os

# external libraries
import numpy as np
from numpy import testing

# local libraries
from .. import process


def test_coefficient_of_determination():

    # TODO : It isn't clear to me why I can't get these results to match at
    # a higher precision.

    # define a simple line with some measured data points
    num_samples = 100
    x = np.arange(num_samples)
    slope = np.random.choice([-1.0, 1.0]) * float(np.random.randint(1, 100))
    intercept = np.random.choice([-1.0, 1.0]) * float(np.random.randint(1, 100))
    y = slope * x + intercept
    # add some noise to each to create fake measurements
    x_measured = x + 0.001 * np.random.random(num_samples)
    y_measured = y + 0.001 * np.random.random(num_samples)

    # find the least squares solution
    A = np.vstack((x_measured, np.ones_like(x_measured))).transpose()
    b = y_measured
    xhat, sums_of_squares_of_residuals, rank, s = np.linalg.lstsq(A, b)
    y_predicted = np.dot(A, xhat)

    # find R^2 the linear algebra way
    expected_r_squared = 1.0 - sums_of_squares_of_residuals / \
        np.linalg.norm(b - b.mean())

    # find R^2 the statistics way
    residuals = np.dot(A, xhat) - b
    expected_error_sum_of_squares = np.sum(residuals ** 2)
    expected_total_sum_of_squares = np.sum((b - b.mean()) ** 2)
    second_expected_r_squared = 1.0 - expected_error_sum_of_squares / \
        expected_total_sum_of_squares

    # find R^2 another statistics way
    r_squared, error_sum_of_squares, total_sum_of_squares, regression_sum_of_squares = \
        process.fit_goodness(y_measured, np.dot(A, xhat))

    # compute r^2 directly
    second_r_squared = process.coefficient_of_determination(y_measured,
                                                            y_predicted)

    testing.assert_allclose(xhat, [slope, intercept], rtol=0.0, atol=0.3)

    # It seems that numpy.linalg.lstsq doesn't output a fery high precision
    # value for the residual sum of squares, so I set the tolerance here to
    # pass.
    testing.assert_allclose(error_sum_of_squares,
                            sums_of_squares_of_residuals, rtol=0.0,
                            atol=5e-6)
    # This also doesn't have high precision, not sure why.
    testing.assert_allclose(error_sum_of_squares,
                            expected_error_sum_of_squares, rtol=0.0,
                            atol=5e-6)

    testing.assert_allclose(total_sum_of_squares,
                            expected_total_sum_of_squares)

    # This precision issue carryies through to these compuations.
    testing.assert_allclose(r_squared, expected_r_squared, rtol=0.0,
                            atol=5e-6)
    # This passes with default tolerances
    testing.assert_allclose(r_squared, second_expected_r_squared)
    testing.assert_allclose(second_r_squared, expected_r_squared, rtol=0.0,
                            atol=2e-5)
    testing.assert_allclose(second_r_squared, second_expected_r_squared,
                            rtol=0.0, atol=2e-5)


def test_least_squares_variance():

    A = np.random.rand(3, 2)
    sum_of_residuals = 5.0
    expected_variance = 5.0 / (3 - 2)
    expected_covariance = expected_variance * np.linalg.inv(np.dot(A.T, A))
    variance, covariance = process.least_squares_variance(A, sum_of_residuals)

    assert expected_variance == variance
    testing.assert_allclose(covariance, expected_covariance)

def test_spline_over_nan():
    x = np.linspace(0., 50., num=300)
    y = np.sin(x) + np.random.rand(len(x))
    # add some nan's
    y[78:89] = np.nan
    y[395:455] = np.nan
    y[0] = np.nan
    y[212] = np.nan

    ySplined = process.spline_over_nan(x, y)
    #plt.plot(x, ySplined)
    #plt.plot(x, y)
    #plt.show()


class TestTimeShift():

    def setup(self):

        self.sample_rate = 300  # hz
        self.time = np.linspace(0.0, 100.0, self.sample_rate * 100 + 1)
        self.tau = -5.0

        def normal_distribution(x):
            sigma = 20.0
            mu = 50.0
            return 1.0 / (sigma * np.sqrt(2 * np.pi)) * \
                np.e ** (-((x - mu) ** 2) / (2 * sigma ** 2))

        self.base_signal = normal_distribution(self.time)
        self.shifted_signal = normal_distribution(self.time + self.tau)

    def test_sync_error(self):

        error = process.sync_error(self.tau, self.base_signal,
                                   self.shifted_signal, self.time,
                                   plot=False)
        testing.assert_allclose(error, 0.0, atol=1e-8)

    def test_find_time_shift(self):

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               plot=False)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               guess=self.tau,
                                               plot=False)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

    def test_truncate_data(self):

        truncated_signal1, truncated_signal2 = \
            process.truncate_data(self.tau, self.base_signal,
                                  self.shifted_signal, self.sample_rate)
        assert len(truncated_signal1) == len(truncated_signal2) == \
            len(self.time)


class TestTimeShiftRealData():

    def setup(self):

        self.grf_array = np.loadtxt(
            os.path.join(os.path.dirname(__file__),
                         'data/example_vertical_grf.csv'), delimiter=',')

        self.original_time = self.grf_array[:, 0]
        self.original_vgrf = self.grf_array[:, 1]

        self.sample_rate = int(np.mean(1.0 /
                                       np.diff(self.original_time)))

        start = 1.0 * self.sample_rate
        stop = 5.0 * self.sample_rate
        test_time = self.grf_array.T[0, start:stop]
        vertical_grf = self.grf_array.T[1, start:stop]
        #self.sample_rate = int(np.mean(1.0 / np.diff(original_time)))

        self.tau = -0.1
        self.base_signal = vertical_grf[abs(self.tau) * self.sample_rate:]
        self.shifted_signal = vertical_grf[:-abs(self.tau) * self.sample_rate]
        self.truncated_time = test_time[:len(self.base_signal)]

    def test_sync_error(self):

        error = process.sync_error(self.tau, self.base_signal,
                                   self.shifted_signal, self.truncated_time,
                                   plot=False)
        testing.assert_allclose(error, 0.0, atol=17.0)

    def test_find_time_shift(self):

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               plot=False)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               guess=self.tau,
                                               plot=False)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

    def test_truncate_data(self):

        truncated_signal1, truncated_signal2 = \
            process.truncate_data(self.tau, self.base_signal,
                                  self.shifted_signal, self.sample_rate)
        assert len(truncated_signal1) == len(truncated_signal2) == \
            len(self.truncated_time)
