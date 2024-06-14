#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os

# external libraries
import numpy as np
from numpy import testing
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
import pytest

# local libraries
from .. import process


def test_derivative():

    x = [0.0, 1.0, 2.0, 3.0, 5.0]
    y = [1.0, 4.0, 9.0, 10.0, 17.0]

    expected_dydx = np.array([3.0, 5.0, 1.0, 3.5])
    dydx = process.derivative(x, y, method='forward')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([0.0, 3.0, 5.0, 1.0, 3.5])
    dydx = process.derivative(x, y, method='backward', padding=0.0)
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([2.0, 4.0, 3.0, 8.0 / 3.0, 5.0])
    dydx = process.derivative(x, y, method='central', padding='second order')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([2.0, 4.0, 3.0, 8.0 / 3.0, 5.0])
    dydx = process.derivative(x, y, method='combination')
    testing.assert_allclose(dydx, expected_dydx)

    y = [[1.0, 2.0],
         [4.0, 5.0],
         [9.0, 10.0],
         [10.0, 11.0],
         [17.0, 18.0]]

    expected_dydx = np.array([[3.0, 3.0],
                              [5.0, 5.0],
                              [1.0, 1.0],
                              [3.5, 3.5]])
    dydx = process.derivative(x, y, method='forward')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[3.0, 3.0],
                              [5.0, 5.0],
                              [1.0, 1.0],
                              [3.5, 3.5],
                              [0.0, 0.0]])
    dydx = process.derivative(x, y, method='forward', padding=0.0)
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[3.0, 3.0],
                              [5.0, 5.0],
                              [1.0, 1.0],
                              [3.5, 3.5],
                              [3.5, 3.5]])
    dydx = process.derivative(x, y, method='forward', padding='adjacent')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[3.0, 3.0],
                              [5.0, 5.0],
                              [1.0, 1.0],
                              [3.5, 3.5]])
    dydx = process.derivative(x, y, method='backward')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[0.0, 0.0],
                              [3.0, 3.0],
                              [5.0, 5.0],
                              [1.0, 1.0],
                              [3.5, 3.5]])
    dydx = process.derivative(x, y, method='backward', padding=0.0)
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[3.0, 3.0],
                              [3.0, 3.0],
                              [5.0, 5.0],
                              [1.0, 1.0],
                              [3.5, 3.5]])
    dydx = process.derivative(x, y, method='backward', padding='adjacent')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[4.0, 4.0],
                              [3.0, 3.0],
                              [8.0 / 3.0, 8.0 / 3.0]])
    dydx = process.derivative(x, y, method='central')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[0.0, 0.0],
                              [4.0, 4.0],
                              [3.0, 3.0],
                              [8.0 / 3.0, 8.0 / 3.0],
                              [0.0, 0.0]])
    dydx = process.derivative(x, y, method='central', padding=0.0)
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[4.0, 4.0],
                              [4.0, 4.0],
                              [3.0, 3.0],
                              [8.0 / 3.0, 8.0 / 3.0],
                              [8.0 / 3.0, 8.0 / 3.0]])

    dydx = process.derivative(x, y, method='central', padding='adjacent')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[2.0, 2.0],
                              [4.0, 4.0],
                              [3.0, 3.0],
                              [8.0 / 3.0, 8.0 / 3.0],
                              [5.0, 5.0]])
    dydx = process.derivative(x, y, method='central', padding='second order')
    testing.assert_allclose(dydx, expected_dydx)

    expected_dydx = np.array([[2.0, 2.0],
                              [4.0, 4.0],
                              [3.0, 3.0],
                              [8.0 / 3.0, 8.0 / 3.0],
                              [5.0, 5.0]])
    dydx = process.derivative(x, y, method='combination')
    testing.assert_allclose(dydx, expected_dydx)

    x = np.linspace(0.0, 10.0, num=1000)
    y = np.vstack((np.sin(x), x**2)).T
    expected_dydx = np.vstack((np.cos(x), 2.0 * x)).T

    dydx = process.derivative(x, y, method='combination')
    testing.assert_allclose(dydx, expected_dydx, rtol=1e-4)

    with pytest.raises(ValueError):
        x = np.ones((5, 2))
        y = np.arange(5)
        process.derivative(x, y)

    with pytest.raises(ValueError):
        x = np.ones(4)
        y = np.ones((5, 3))
        process.derivative(x, y)

    with pytest.raises(ValueError):
        x = np.ones(5)
        y = np.ones((5, 3, 2))
        process.derivative(x, y)

    with pytest.raises(ValueError):
        x = np.ones(1)
        y = np.ones((1, 3))
        process.derivative(x, y)

    with pytest.raises(ValueError):
        x = np.ones(2)
        y = np.ones((2, 3))
        process.derivative(x, y, method='central')

    with pytest.raises(NotImplementedError):
        x = np.ones(2)
        y = np.ones((2, 3))
        process.derivative(x, y, method='booger')


def test_butterworth():

    # setup
    time = np.linspace(0.0, 1.0, 2001)
    sample_rate = 1.0 / np.diff(time).mean()

    low_freq = np.sin(5.0 * 2.0 * np.pi * time)  # 5 Hz * 2 pi rad / cycle
    high_freq = np.sin(250.0 * 2.0 * np.pi * time)  # 250 Hz * 2 pi rad / cycle

    # shape(n,)
    filtered = process.butterworth(low_freq + high_freq, 125.0, sample_rate,
                                   order=8, axis=0, padlen=150)

    testing.assert_allclose(filtered, low_freq, rtol=3e-5, atol=3e-5)

    # shape(n,2)
    data = np.vstack((low_freq + high_freq, low_freq + high_freq)).T

    filtered = process.butterworth(data, 125.0, sample_rate, order=8,
                                   axis=0, padlen=150)

    expected = np.vstack((low_freq, low_freq)).T

    testing.assert_allclose(filtered, expected, rtol=3e-5, atol=3e-5)

    # shape(2,n)
    data = np.vstack((low_freq + high_freq, low_freq + high_freq))

    filtered = process.butterworth(data, 125.0, sample_rate, order=8,
                                   padlen=150)

    expected = np.vstack((low_freq, low_freq))

    testing.assert_allclose(filtered, expected, rtol=1e-5, atol=1e-3)


def test_butterworth_white_noise(plot=False):
    sample_rate = 1000  # Hz
    duration = 10.0  # seconds
    time = np.linspace(0.0, duration, num=int(sample_rate*duration) + 1)
    white_noise = np.random.normal(0.0, 1.0, size=len(time))
    cutoff = 200  # Hz
    order = 4
    double_pass_filtered = process.butterworth(white_noise, cutoff,
                                               sample_rate, btype='lowpass',
                                               order=order)
    sos = butter(order, cutoff/sample_rate*2, btype='lowpass', output='sos')
    single_pass_filtered = sosfilt(sos, white_noise)

    if plot:
        freq, amp = process.freq_spectrum(white_noise, sample_rate)
        freq_double, amp_double = process.freq_spectrum(double_pass_filtered,
                                                        sample_rate)
        freq_single, amp_single = process.freq_spectrum(single_pass_filtered,
                                                        sample_rate)
        fig, ax = plt.subplots()
        ax.plot(freq, amp, label='Unfiltered')
        ax.plot(freq_single, amp_single, alpha=0.75,
                label='Single Pass Filtered')
        ax.plot(freq_double, amp_double, alpha=0.5,
                label='Double Pass Filtered')
        ax.axvline(cutoff, color='black')
        ax.set_ylabel('Amplitude of White Noise with STD=1')
        ax.set_xlabel('Frequency [Hz]')
        msg = 'Sample rate: {} Hz, Cutoff: {} Hz, Order: {}'
        ax.set_title(msg.format(sample_rate, cutoff, order))
        ax.legend()
        plt.show()


def test_coefficient_of_determination():

    # TODO : It isn't clear to me why I can't get these results to match at
    # a higher precision.

    # define a simple line with some measured data points
    num_samples = 100
    x = np.arange(num_samples)
    slope = np.random.uniform(-100, 100)
    intercept = np.random.uniform(-100, 100)
    y = slope * x + intercept
    # add some noise to each to create fake measurements
    x_measured = x + 0.001 * np.random.random(num_samples)
    y_measured = y + 0.001 * np.random.random(num_samples)

    # find the least squares solution
    A = np.vstack((x_measured, np.ones_like(x_measured))).transpose()
    b = y_measured
    # NOTE : Set because of this FutureWarning:
    # ........../home/moorepants/src/DynamicistToolKit/dtk/test/test_process.py:241: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    # To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
    # xhat, sums_of_squares_of_residuals, rank, s = np.linalg.lstsq(A, b)
    xhat, sums_of_squares_of_residuals, rank, s = np.linalg.lstsq(A, b,
                                                                  rcond=None)
    y_predicted = np.dot(A, xhat)

    # find R^2 the linear algebra way
    expected_r_squared = (1.0 - sums_of_squares_of_residuals /
                          np.linalg.norm(b - b.mean()) ** 2)

    # find R^2 the statistics way
    residuals = np.dot(A, xhat) - b
    expected_error_sum_of_squares = np.sum(residuals ** 2)
    expected_total_sum_of_squares = np.sum((b - b.mean()) ** 2)
    second_expected_r_squared = (1.0 - expected_error_sum_of_squares /
                                 expected_total_sum_of_squares)

    # find R^2 another statistics way
    (r_squared, error_sum_of_squares, total_sum_of_squares,
     regression_sum_of_squares) = process.fit_goodness(y_measured,
                                                       np.dot(A, xhat))

    # compute r^2 directly
    second_r_squared = process.coefficient_of_determination(y_measured,
                                                            y_predicted)

    testing.assert_allclose(xhat, [slope, intercept], rtol=0.0, atol=0.3)

    # It seems that numpy.linalg.lstsq doesn't output a very high precision
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
    testing.assert_allclose(r_squared, expected_r_squared)
    testing.assert_allclose(r_squared, second_expected_r_squared)
    testing.assert_allclose(second_r_squared, expected_r_squared)
    testing.assert_allclose(second_r_squared, second_expected_r_squared)


def test_least_squares_variance():

    A = np.random.rand(3, 2)
    sum_of_residuals = 5.0
    expected_variance = 5.0 / (3 - 2)
    expected_covariance = expected_variance * np.linalg.inv(np.dot(A.T, A))
    variance, covariance = process.least_squares_variance(A, sum_of_residuals)

    assert expected_variance == variance
    testing.assert_allclose(covariance, expected_covariance)


def test_spline_over_nan(plot=False):
    x = np.linspace(0., 50., num=300)
    y = np.sin(x) + np.random.rand(len(x))
    # add some nan's
    y[78:89] = np.nan
    y[395:455] = np.nan
    y[0] = np.nan
    y[212] = np.nan

    ySplined = process.spline_over_nan(x, y)

    if plot:
        plt.plot(x, ySplined)
        plt.plot(x, y)
        plt.show()


class TestTimeShift():

    def setup_method(self):

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

    def setup_method(self):

        self.grf_array = np.loadtxt(
            os.path.join(os.path.dirname(__file__),
                         'data/example_vertical_grf.csv'), delimiter=',')

        self.original_time = self.grf_array[:, 0]
        self.original_vgrf = self.grf_array[:, 1]

        self.sample_rate = int(np.mean(1.0 /
                                       np.diff(self.original_time)))

        start = 1 * self.sample_rate
        stop = 5 * self.sample_rate
        test_time = self.grf_array.T[0, start:stop]
        vertical_grf = self.grf_array.T[1, start:stop]
        # self.sample_rate = int(np.mean(1.0 / np.diff(original_time)))

        self.tau = -0.1
        self.base_signal = vertical_grf[int(abs(self.tau) * self.sample_rate):]
        self.shifted_signal = vertical_grf[:-int(abs(self.tau) * self.sample_rate)]
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


def test_time_vector():

    expected_time = [0.0, 1.0, 2.0, 3.0, 4.0]
    testing.assert_allclose(process.time_vector(5, 1.0), expected_time)

    expected_time = [1.0, 2.0, 3.0, 4.0, 5.0]
    testing.assert_allclose(process.time_vector(5, 1.00, 1.0), expected_time)


class TestSpectralAnalysis:
    """
    Test the functions freq_spectrum(), power_spectrum() and 
    cumulative_power_spectrum() based on a common test signal.
    
    """

    def setup_method(self):
        """
        Create test signal.

        """
        # test time and sampling rate. Make sure that the number of samples is
        # a power of two to prevent frequency leakage (freq_spectrum does zero-
        # padding)
        T = 5.12  # signal period
        self.f_s = 100  # sample rate

        t = np.arange(0, T, 1/self.f_s)
        N = len(t)

        # test function
        def harmonics(t, frequencies, amplitudes):
            """
            Create a test function with three harmonic frequencies.

            x(t) = A0 * sin(2*pi*f0*t) + ... + An * sin(2*pi*fn*t)

            Parameters
            ----------
            t : array-like
                Array of sample times
            frequencies : array-like
                Array of n frequencies
            amplitudes : array-like
                Array of n amplitudes

            Returns
            -------
            x : array-like
                Array of samples of the signal.
            """
            x = np.zeros_like(t, dtype=float)
            for f, A in zip(frequencies, amplitudes):
                x += A * np.sin(2*np.pi*f*t)
            return x

        # Build a test signal of three harmonics. The frequencies have to be
        # multiples of the frequency resolution to prevent frequency leakage.
        self.frequencies_k = np.array((5, 12, 17, 28, 25, 40))
        self.frequencies = self.frequencies_k * self.f_s / N
        self.amplitudes = np.array((0.5, 0.9, 0.4, 0.3, 0.6, 0.1))

        self.x = harmonics(t, self.frequencies, self.amplitudes)

    def test_freq_spectrum(self):

        # call frequency spectrum
        freq, amp = process.freq_spectrum(self.x, self.f_s)

        # build expected spectrum. This should have discrete peaks at the
        # chosen frequencies with the designed amplitude of the harmonic signal
        # frequs.
        amp_exp = np.zeros_like(amp)
        for fk, a in zip(self.frequencies_k, self.amplitudes):
            amp_exp[fk-1] = a  # shift by 1, bc freq_spectrum skips f=0

        # plot the results for inspection
        # plt.plot(freq, amp   , "r.")
        # plt.plot(freq, amp_exp, "b.")

        msg = ("The result freq_spectrum does "
               "not match the theoretical expectation.")
        assert np.allclose(amp, amp_exp), msg

    def test_power_spectrum(self):

        # call power spectrum
        freq, amp = process.power_spectrum(self.x, self.f_s)

        # build expected spectrum. This should have discrete peaks at the
        # chosen frequencies. The theoretical mean power of a sine over one
        # period is A**2 / 2.
        amp_exp = np.zeros_like(amp)
        for fk, a in zip(self.frequencies_k, self.amplitudes):
            amp_exp[fk] = a**2 / 2

        # plot the results for inspection
        # plt.plot(freq, amp   , "r.")
        # plt.plot(freq, amp_exp, "b.")

        # check the power spectral density
        msg = ("The result of freq_spectrum() does not match the theoretical "
               "expectation.")
        assert np.allclose(amp, amp_exp), msg

        # check Parseval's theorem
        avgpwr_time = np.mean(np.abs(self.x)**2)
        avgpwr_freq = np.sum(amp)

        msg = ("The result of freq_spectrum() does not satisfy Parseval's "
               "theorem!")
        assert np.allclose(avgpwr_time, avgpwr_freq), msg

    def test_cumulative_power_spectrum(self):

        # Absolute case:

        # call cumulative power spectrum
        freq, amp = process.cumulative_power_spectrum(self.x, 
                                                    self.f_s, 
                                                    relative=False)

        # build expected spectrum. This is the cumulative version of the
        # spectrum used in test_power_spectrum()
        amp_exp = np.zeros_like(amp)
        for fk, a in zip(self.frequencies_k, self.amplitudes):
            amp_exp[fk] = a**2 / 2
        amp_exp = np.cumsum(amp_exp)

        # plt.plot(freq, amp   , "r.")
        # plt.plot(freq, amp_exp, "b.")

        # check the power spectral density
        msg = ("The result of freq_spectrum() does not match the theoretical "
               "expectation for the absolute case.")
        assert np.allclose(amp, amp_exp), msg

        # Relative case:

        # call cumulative power spectrum with relative = True (default)
        freq, amp = process.cumulative_power_spectrum(self.x, self.f_s)

        # normalize the  spectrum.
        amp_exp = amp_exp / amp_exp[-1]

        # check the power spectral density
        msg = ("The result of freq_spectrum() does not match the theoretical "
               "expectation for the relative case.")
        assert np.allclose(amp, amp_exp), msg
