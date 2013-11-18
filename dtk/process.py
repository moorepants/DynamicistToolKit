#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external dependencies
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin
from scipy.signal import butter, filtfilt
from scipy.stats import nanmean
from scipy import sparse
import matplotlib.pyplot as plt


def sync_error(tau, signal1, signal2, time, plot=False):
    '''Returns the error between two signal time histories given a time
    shift, tau.

    Parameters
    ----------
    tau : float
        The time shift.
    signal1 : ndarray, shape(n,)
        The signal that will be interpolated. This signal is
        typically "cleaner" that signal2 and/or has a higher sample rate.
    signal2 : ndarray, shape(n,)
        The signal that will be shifted to syncronize with signal 1.
    time : ndarray, shape(n,)
        The time vector for the two signals
    plot : boolean, optional, default=False
        If true a plot will be shown of the resulting signals.

    Returns
    -------
    error : float
        Error between the two signals for the given tau.

    '''
    # make sure tau isn't too large
    if np.abs(tau) >= time[-1]:
        raise ValueError(('abs(tau), {0}, must be less than or equal to ' +
                         '{1}').format(str(np.abs(tau)), str(time[-1])))

    # this is the time for the second signal which is assumed to lag the first
    # signal
    shiftedTime = time + tau

    # create time vector where the two signals overlap
    if tau > 0:
        intervalTime = shiftedTime[np.nonzero(shiftedTime < time[-1])]
    else:
        intervalTime = shiftedTime[np.nonzero(shiftedTime > time[0])]

    # interpolate between signal 1 samples to find points that correspond in
    # time to signal 2 on the shifted time
    sig1OnInterval = np.interp(intervalTime, time, signal1)

    # truncate signal 2 to the time interval
    if tau > 0:
        sig2OnInterval = signal2[np.nonzero(shiftedTime <= intervalTime[-1])]
    else:
        sig2OnInterval = signal2[np.nonzero(shiftedTime >= intervalTime[0])]

    if plot is True:
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(time, signal1, time, signal2)
        axes[0].legend(('Signal 1', 'Signal 2'))
        axes[0].set_title("Before shifting.")
        axes[1].plot(intervalTime, sig1OnInterval, intervalTime,
                     sig2OnInterval)
        axes[1].set_title("After shifting.")
        axes[1].legend(('Signal 1', 'Signal 2'))
        plt.show()

    # calculate the error between the two signals
    error = np.linalg.norm(sig1OnInterval - sig2OnInterval)

    return error


def find_timeshift(signal1, signal2, sample_rate, guess=None, plot=False):
    '''Returns the timeshift, tau, of the second signal relative to the
    first signal.

    Parameters
    ----------
    signal1 : array_like, shape(n, )
        The base signal.
    signal2 : array_like, shape(n, )
        A signal shifted relative to the first signal. The second signal
        should be leading the first signal.
    sample_rate : integer or float
        Sample rate of the signals. This should be the same for each signal.
    guess : float, optional, default=None
        If you've got a good guess for the time shift then supply it here.
    plot : boolean, optional, defaul=False
        If true, a plot of the error landscape will be shown.

    Returns
    -------
    tau : float
        The timeshift between the two signals.

    '''
    # raise an error if the signals are not the same length
    if len(signal1) != len(signal2):
        raise ValueError('Signals are not the same length!')

    # subtract the mean and normalize both signals
    signal1 = normalize(subtract_mean(signal1))
    signal2 = normalize(subtract_mean(signal2))

    time = time_vector(len(signal1), sample_rate)

    if guess is None:
        # set up the error landscape, error vs tau
        # We assume the time shift is
        tau_range = np.linspace(-time[len(time) / 4], time[len(time) / 4],
                                num=len(time) / 10)

        # TODO : Can I vectorize this?
        error = np.zeros_like(tau_range)
        for i, val in enumerate(tau_range):
            error[i] = sync_error(val, signal1, signal2, time)

        if plot is True:
            plt.figure()
            plt.plot(tau_range, error)
            plt.xlabel('tau')
            plt.ylabel('error')
            plt.show()

        # find initial condition from landscape
        tau0 = tau_range[np.argmin(error)]
    else:
        tau0 = guess


    print "The minimun of the error landscape is {}.".format(tau0)

    tau, fval = fmin(sync_error, tau0, args=(signal1, signal2, time),
                     full_output=True, disp=True)[0:2]

    return tau


def truncate_data(tau, signal1, signal2, sample_rate):
    '''Returns the truncated vectors with respect to the time shift tau. It
    assume you've found the time shift between two signals with
    find_time_shift or something similar.

    Parameters
    ----------
    tau : float
        The time shift.
    signal1 : array_like, shape(n, )
        A time series.
    signal2 : array_like, shape(n, )
        A time series.
    sample_rate : integer
        The sample rate of the two signals.

    Returns
    -------
    truncated1 : ndarray, shape(m, )
        The truncated time series.
    truncated2 : ndarray, shape(m, )
        The truncated time series.

    '''
    t = time_vector(len(signal1), sample_rate)

    # shift the first signal
    t1 = t - tau
    t2 = t

    # make the common time interval
    common_interval = t2[np.nonzero(t2 < t1[-1])]

    truncated1 = np.interp(common_interval, t1, signal1)
    truncated2 = signal2[np.nonzero(t2 <= common_interval[-1])]

    return truncated1, truncated2


def least_squares_variance(A, sum_of_residuals):
    """Returns the variance in the ordinary least squares fit and the
    covariance matrix of the estimated parameters.

    Parameters
    ----------
    A : ndarray, shape(n,d)
        The left hand side matrix in Ax=B.
    sum_of_residuals : float
        The sum of the residuals (residual sum of squares).

    Returns
    -------
    variance : float
        The variance of the fit.
    covariance : ndarray, shape(d,d)
        The covariance of x in Ax = b.

    """
    # I am pretty sure that the residuals from numpy.linalg.lstsq is the SSE
    # (the residual sum of squares).

    degrees_of_freedom = (A.shape[0] - A.shape[1])
    variance = sum_of_residuals / degrees_of_freedom

    # There may be a way to use the pinv here for more efficient
    # computations. (A^T A)^-1 A^T = np.linalg.pinv(A) so np.linalg.pinv(A)
    # (A^T)^-1 ... or maybe not.
    if sparse.issparse(A):
        inv = sparse.linalg.inv
        prod = A.T * A
    else:
        inv = np.linalg.inv
        prod = np.dot(A.T, A)

    covariance = variance * inv(prod)

    return variance, covariance


def coefficient_of_determination(measured, predicted):
    """Computes the coefficient of determination with respect to a measured
    and predicted array.

    Parameters
    ----------
    measured : array_like, shape(n,)
        The observed or measured values.
    predicted : array_like, shape(n,)
        The values predicted by a model.

    Returns
    -------
    r_squared : float
       The coefficient of determination.

    Notes
    -----

    The coefficient of determination (also referred to as R^2 and VAF
    (variance accounted for) is computed either of these two ways::

            sum( [predicted - mean(measured)] ** 2 )
      R^2 = ----------------------------------------
            sum( [measured - mean(measured)] ** 2 )

    or::

                sum( [measured - predicted] ** 2 )
      R^2 = 1 - ---------------------------------------
                sum( [measured - mean(measured)] ** 2 )


    """

    numerator = np.linalg.norm(measured - predicted)
    denominator = np.linalg.norm(measured - measured.mean())

    r_squared = 1.0 - numerator / denominator

    return r_squared


def fit_goodness(ym, yp):
    '''
    Calculate the goodness of fit.

    Parameters
    ----------
    ym : ndarray, shape(n,)
        The vector of measured values.
    yp : ndarry, shape(n,)
        The vector of predicted values.

    Returns
    -------
    rsq : float
        The r squared value of the fit.
    SSE : float
        The error sum of squares.
    SST : float
        The total sum of squares.
    SSR : float
        The regression sum of squares.

    Notes
    -----

    SST = SSR + SSE

    '''

    ym_bar = np.mean(ym)
    SSR = sum((yp - ym_bar) ** 2)
    SST = sum((ym - ym_bar) ** 2)
    SSE = SST - SSR
    rsq = SSR / SST

    return rsq, SSE, SST, SSR


def spline_over_nan(x, y):
    """
    Returns a vector of which a cubic spline is used to fill in gaps in the
    data from nan values.

    Parameters
    ----------
    x : ndarray, shape(n,)
        This x values should not contain nans.
    y : ndarray, shape(n,)
        The y values may contain nans.

    Returns
    -------
    ySpline : ndarray, shape(n,)
        The splined y values. If `y` doesn't contain any nans then `ySpline` is
        `y`.

    Notes
    -----
    The splined data is identical to the input data, except that the nan's are
    replaced by new data from the spline fit.

    """

    # if there are nans in the data then spline away
    if np.isnan(y).any():
        # remove the values with nans
        xNoNan = x[np.nonzero(np.isnan(y) == False)]
        yNoNan = y[np.nonzero(np.isnan(y) == False)]
        # fit a spline through the data
        spline = UnivariateSpline(xNoNan, yNoNan, k=3, s=0)
        return spline(x)
    else:
        return y


def curve_area_stats(x, y):
    '''
    Return the box plot stats of a curve based on area.

    Parameters
    ----------
    x : ndarray, shape (n,)
        The x values
    y : ndarray, shape (n,m)
        The y values
        n are the time steps
        m are the various curves

    Returns
    -------
    A dictionary containing:
    median : ndarray, shape (m,)
        The x value corresponding to 0.5*area under the curve
    lq : ndarray, shape (m,)
        lower quartile
    uq : ndarray, shape (m,)
        upper quartile
    98p : ndarray, shape (m,)
        98th percentile
    2p : ndarray, shape (m,)
        2nd percentile

    '''
    area = trapz(y, x=x, axis=0) # shape (m,)
    percents = np.array([0.02*area, 0.25*area, 0.5*area, 0.75*area, 0.98*area]) # shape (5,m)
    CumArea = cumtrapz(y.T, x=x.T) # shape(m,n)
    xstats = {'2p':[], 'lq':[], 'median':[], 'uq':[], '98p':[]}
    for j, curve in enumerate(CumArea):
        flags = [False for flag in range(5)]
        for i, val in enumerate(curve):
            if val > percents[0][j] and flags[0] == False:
                xstats['2p'].append(x[i])
                flags[0] = True
            elif val > percents[1][j] and flags[1] == False:
                xstats['lq'].append(x[i])
                flags[1] = True
            elif val > percents[2][j] and flags[2] == False:
                xstats['median'].append(x[i])
                flags[2] = True
            elif val > percents[3][j] and flags[3] == False:
                xstats['uq'].append(x[i])
                flags[3] = True
            elif val > percents[4][j] and flags[4] == False:
                xstats['98p'].append(x[i])
                flags[4] = True
        if flags[4] == False:
            # this is what happens if it finds none of the above
            xstats['2p'].append(0.)
            xstats['lq'].append(0.)
            xstats['median'].append(0.)
            xstats['uq'].append(0.)
            xstats['98p'].append(0.)
    for k, v in xstats.items():
        xstats[k] = np.array(v)
    return xstats


def freq_spectrum(data, sampleRate):
    """
    Return the frequency spectrum of a data set.

    Parameters
    ----------
    data : ndarray, shape (m,) or shape(n,m)
        The array of time signals where n is the number of variables and m is
        the number of time steps.
    sampleRate : int
        The signal sampling rate in hertz.

    Returns
    -------
    frequency : ndarray, shape (p,)
        The frequencies where p is a power of 2 close to m.
    amplitude : ndarray, shape (p,n)
        The amplitude at each frequency.

    """
    def nextpow2(i):
        '''
        Return the next power of 2 for the given number.

        '''
        n = 2
        while n < i: n *= 2
        return n

    time = 1. / sampleRate # sample time
    try:
        L = data.shape[1] # length of data if (n, m)
    except:
        L = data.shape[0] # length of data if (n,)
    # calculate the closest power of 2 for the length of the data
    n = nextpow2(L)
    Y = fft(data, n) / L # divide by L for scaling
    f = fftfreq(n, d=time)
    #f = sampleRate/2.*linspace(0, 1, n)
    #print 'f =', f, f.shape, type(f)
    frequency = f[1:n / 2]
    try:
        amplitude = 2 * abs(Y[:, 1:n / 2]).T # multiply by 2 because we take half the vector
        #power = abs(Y[:, 1:n/2])**2
    except:
        amplitude = 2 * abs(Y[1:n / 2])
        #power = abs(Y[1:n/2])**2
    return frequency, amplitude


def butterworth(data, cutoff, sampleRate, order=2, axis=-1):
    """
    Returns the filtered data for a low pass Butterworth filter.

    Parameters
    ----------
    data : ndarray
        The data to filter.
    cutoff : float or int
        The cutoff frequency in hertz.
    sampleRate : float or int
        The sampling rate in hertz.
    order : int
        The order of the Butterworth filter.
    axis : int
        The axis to filter along.

    Returns
    -------
    filteredData : ndarray
        The low pass filtered version of data.

    Notes
    -----
    This does a forward and backward Butterworth filter.

    """

    b, a = butter(order, float(cutoff) / float(sampleRate) / 2.)

    return filtfilt(b, a, data, axis=axis)


def subtract_mean(sig, hasNans=False):
    '''
    Subtracts the mean from a signal with nanmean.

    Parameters
    ----------
    sig : ndarray, shape(n,)
    hasNans : boolean, optional
        If your data has nans use this flag if you want to ignore them.

    Returns
    -------
    ndarray, shape(n,)
        sig minus the mean of sig

    '''
    if hasNans:
        return sig - nanmean(sig)
    else:
        return sig - np.mean(sig)


def normalize(sig, hasNans=False):
    '''
    Normalizes the vector with respect to the maximum value.

    Parameters
    ----------
    sig : ndarray, shape(n,)
    hasNans : boolean, optional
        If your data has nans use this flag if you want to ignore them.

    Returns
    -------
    normSig : ndarray, shape(n,)
        The signal normalized with respect to the maximum value.

    '''
    # TODO : This could be a try/except statement instead of an optional
    # argument.
    if hasNans:
        normSig = sig / np.nanmax(sig)
    else:
        normSig = sig / np.max(sig)

    return normSig


def derivative(x, y, method='forward'):
    '''Returns the derivative of y with respect to x.

    Parameters
    ----------
    x : ndarray, shape(n,)
    y : ndarray, shape(n,)
    method : string, optional
        'forward'
           Use the forward difference method.
        'central'
          Use the central difference method.
        'backward'
          Use the backward difference method.
        'combination'
          Use forward on the first point, backward on the last and central
          on the rest.

    Returns
    -------
    dydx : ndarray, shape(n,) or shape(n-1,)
        for combination else shape(n-1,)

    '''
    if method == 'forward':
        return np.diff(y) / np.diff(x)
    elif method == 'combination':
        dxdy = np.zeros_like(y)
        for i, yi in enumerate(y[:]):
            if i == 0:
                dxdy[i] = (-3 * y[0] + 4 * y[1] - y[2])\
                          / 2 / (x[1] - x[0])
            elif i == len(y) - 1:
                dxdy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3])\
                           / 2 / (x[-1] - x[-2])
            else:
                dxdy[i] = (y[i + 1] - y[i - 1]) / 2 / (x[i] - x[i - 1])
        return dxdy
    else:
        raise NotImplementedError("There is no %s method here! Only 'forward'\
            and 'combination' are currently available." % method)


def time_vector(numSamples, sampleRate):
    '''Returns a time vector starting at zero.

    Parameters
    ----------
    numSamples : int or float
        Total number of samples.
    sampleRate : int or float
        Sample rate of the signal in hertz.

    Returns
    -------
    time : ndarray, shape(numSamples,)
        Time vector starting at zero.

    '''

    ns = float(numSamples)
    sr = float(sampleRate)

    return np.linspace(0., (ns - 1.) / sr, num=ns)
