#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from dtk import process
import numpy as np
from numpy import testing


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
                                   plot=True)
        testing.assert_allclose(error, 0.0, atol=1e-8)

    def test_find_time_shift(self):

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               plot=True)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               guess=self.tau,
                                               plot=True)
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
                                   plot=True)
        testing.assert_allclose(error, 0.0, atol=17.0)

    def test_find_time_shift(self):

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               plot=True)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

        estimated_tau = process.find_timeshift(self.base_signal,
                                               self.shifted_signal,
                                               self.sample_rate,
                                               guess=self.tau,
                                               plot=True)
        testing.assert_allclose(estimated_tau, self.tau, atol=0.1)

    def test_truncate_data(self):

        truncated_signal1, truncated_signal2 = \
            process.truncate_data(self.tau, self.base_signal,
                                  self.shifted_signal, self.sample_rate)
        assert len(truncated_signal1) == len(truncated_signal2) == \
            len(self.truncated_time)
