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


def test_sync_error(plot=False):
    sample_rate = 300  # hz
    time = np.linspace(0.0, 100.0, sample_rate * 100 + 1)
    tau = -5.0

    def normal_distribution(x):
        sigma = 20.0
        mu = 50.0
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * \
            np.e ** (-((x - mu) ** 2) / (2 * sigma ** 2))

    base_signal = normal_distribution(time)
    shifted_signal = normal_distribution(time + tau)

    error = process.sync_error(tau, base_signal, shifted_signal, time,
                               plot=plot)
    testing.assert_allclose(error, 0.0, atol=1e-8)

    grf_array = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                        'data/example_vertical_grf.csv'),
                           delimiter=',')

    tau = -0.1
    start = 4.0 * sample_rate
    stop = 5.0 * sample_rate
    original_time = grf_array.T[0, start:stop]
    vertical_grf = grf_array.T[1, start:stop]
    sample_rate = int(np.mean(1.0 / np.diff(original_time)))

    base_signal = vertical_grf[abs(tau) * sample_rate:]
    shifted_signal = vertical_grf[:-abs(tau) * sample_rate]
    truncated_time = original_time[:len(base_signal)]

    error = process.sync_error(tau, base_signal, shifted_signal,
                               truncated_time, plot=plot)
    testing.assert_allclose(error, 0.0, atol=13.0)
