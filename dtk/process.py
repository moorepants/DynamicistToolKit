#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external dependencies
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin
from scipy.signal import butter, sosfiltfilt
try:
    from scipy.stats import nanmean
except ImportError:  # NOTE : nanmean was removed from SciPy in version 0.18.0.
    from numpy import nanmean
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

    Examples
    --------

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import find_timeshift, sync_error

       t = np.linspace(0.0, 4.0, num=401)
       sig1 = np.sin(2.0*t) + np.random.normal(0.0, 0.1, size=len(t))
       sig2 = np.sin(2.0*t + 0.3) + np.random.normal(0.0, 0.1, size=len(t))

       tau = find_timeshift(sig1, sig2, 100, guess=0.2)

       sync_error(tau, sig1, sig2, t, plot=True)

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
        fig, axes = plt.subplots(2, 1, layout='constrained')
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
    '''Returns the timeshift, tau, of the second signal relative to the first
    signal.

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

    Examples
    --------

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import find_timeshift

       t = np.linspace(0.0, 4.0, num=401)
       sig1 = np.sin(2.0*t) + np.random.normal(0.0, 0.1, size=len(t))
       sig2 = np.sin(2.0*t + 0.3) + np.random.normal(0.0, 0.1, size=len(t))

       tau = find_timeshift(sig1, sig2, 100, guess=0.2)

       fig, ax = plt.subplots(layout='constrained')
       ax.plot(t, sig1, t, sig2)
       ax.legend(['Signal 1', 'Signal 2'])
       ax.set_title('Shift: {:1.3f} s'.format(tau))
       ax.set_xlabel('Time [s]')

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
        tau_range = np.linspace(-time[len(time) // 4], time[len(time) // 4],
                                num=len(time) // 10)

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

    res = fmin(sync_error, tau0, args=(signal1, signal2, time))

    return res[0]


def truncate_data(tau, signal1, signal2, sample_rate):
    '''Returns the truncated vectors with respect to the time shift tau. It
    assume you've found the time shift between two signals with find_time_shift
    or something similar.

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

    Examples
    --------

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import find_timeshift, truncate_data

       t = np.linspace(0.0, 4.0, num=401)
       sig1 = np.sin(2.0*t) + np.random.normal(0.0, 0.1, size=len(t))
       sig2 = np.sin(2.0*t + 0.3) + np.random.normal(0.0, 0.1, size=len(t))

       tau = find_timeshift(sig1, sig2, 100, guess=0.2)

       sigtr1, sigtr2 = truncate_data(tau, sig1, sig2, 100)

       fig, ax = plt.subplots(layout='constrained')
       ax.plot(t[:len(sigtr1)], sigtr1, t[:len(sigtr2)], sigtr2)
       ax.legend(['Signal 1', 'Signal 2'])
       ax.set_title('Shift: {:1.3f} s'.format(tau))
       ax.set_xlabel('Time [s]')

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

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.process import coefficient_of_determination
    >>> np.random.seed(10)
    >>> t = np.linspace(0.0, 10.0, num=1001)
    >>> predicted = np.sin(t)
    >>> measured = predicted + np.random.normal(0.01, 0.1, size=len(t))
    >>> coefficient_of_determination(measured, predicted)
    0.980225686442542

    Notes
    -----

    The coefficient of determination [also referred to as R^2 and VAF
    (variance accounted for)] is computed either of these two ways::

            sum( [predicted - mean(measured)] ** 2 )
      R^2 = ----------------------------------------
            sum( [measured - mean(measured)] ** 2 )

    or::

                sum( [measured - predicted] ** 2 )
      R^2 = 1 - ---------------------------------------
                sum( [measured - mean(measured)] ** 2 )


    """
    # 2-norm => np.sqrt(np.sum(measured - predicted)**2))

    numerator = np.linalg.norm(measured - predicted)**2
    denominator = np.linalg.norm(measured - measured.mean())**2

    r_squared = 1.0 - numerator/denominator

    # TODO : Does not give the same r^2 as fit_goodness below.

    return float(r_squared)


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

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.process import fit_goodness
    >>> np.random.seed(10)
    >>> t = np.linspace(0.0, 10.0, num=1001)
    >>> predicted = np.sin(t)
    >>> measured = predicted + np.random.normal(0.01, 0.1, size=len(t))
    >>> fit_goodness(measured, predicted)
    (0.9862246380225299, 6.197685397793293, 449.9108922095648, 443.7132068117715)

    Notes
    -----

    SST = SSR + SSE

    '''

    ym_bar = np.mean(ym)
    SSR = sum((yp - ym_bar)**2)
    SST = sum((ym - ym_bar)**2)
    SSE = SST - SSR
    rsq = SSR/SST

    return float(rsq), float(SSE), float(SST), float(SSR)


def spline_over_nan(x, y):
    """Returns a vector of which a cubic spline is used to fill in gaps in the
    data from nan values.

    Parameters
    ----------
    x : array_like, shape(n,)
        This x values should not contain nans.
    y : array_like, shape(n,)
        The y values may contain nans.

    Returns
    -------
    ndarray, shape(n,)
        The splined y values. If `y` doesn't contain any nans the ``y`` is
        returned unmodified.

    Notes
    -----

    The splined data is identical to the input data, except that the nan's are
    replaced by new data from the spline fit.

    Examples
    --------

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       from dtk.process import spline_over_nan

       x = np.linspace(0.0, 2.0, num=201)
       y = np.sin(3*2*np.pi*x) + np.random.normal(0.0, 0.1, size=len(x))

       y[78:89] = np.nan
       y[95:102] = np.nan
       y[189:192] = np.nan

       y_splined = spline_over_nan(x, y)

       fig, ax = plt.subplots(layout='constrained')
       ax.fill_between(x, np.min(y_splined) - 0.5, np.max(y_splined) + 0.5,
                       where=np.isnan(y), alpha=0.5, color='grey',
                       transform=ax.get_xaxis_transform())
       ax.plot(x, y_splined, linewidth=4, color='black', label='Filled')
       ax.plot(x, y, linestyle='', marker='o', color='red', label='Missing')
       ax.legend()

    """

    # if there are nans in the data then spline away
    if np.isnan(y).any():
        # remove the values with nans
        x_no_nan = x[np.nonzero(np.isnan(y) == False)]
        y_no_nan = y[np.nonzero(np.isnan(y) == False)]
        # fit a spline through the data
        spline = UnivariateSpline(x_no_nan, y_no_nan, k=3, s=0)
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

    Examples
    --------

    >>> from pprint import pprint
    >>> import numpy as np
    >>> from dtk.process import curve_area_stats
    >>> x = np.linspace(0.0, 10.0, num=1001)
    >>> y = np.vstack((np.exp(x), 0.5*x)).T
    >>> pprint(curve_area_stats(x, y))
    {'2p': array([6.09, 1.41]),
     '98p': array([9.97, 9.89]),
     'lq': array([8.61, 5.  ]),
     'median': array([9.3 , 7.07]),
     'uq': array([9.71, 8.66])}

    '''
    area = trapezoid(y, x=x, axis=0)  # shape (m,)
    percents = np.array([0.02*area,
                         0.25*area,
                         0.5*area,
                         0.75*area,
                         0.98*area])  # shape (5,m)

    cummlative_area = cumulative_trapezoid(y.T, x=x.T)  # shape(m,n)

    xstats = {'2p': [], 'lq': [], 'median': [], 'uq': [], '98p': []}
    for j, curve in enumerate(cummlative_area):
        flags = [False for flag in range(5)]
        for i, val in enumerate(curve):
            if val > percents[0][j] and not flags[0]:
                xstats['2p'].append(x[i])
                flags[0] = True
            elif val > percents[1][j] and not flags[1]:
                xstats['lq'].append(x[i])
                flags[1] = True
            elif val > percents[2][j] and not flags[2]:
                xstats['median'].append(x[i])
                flags[2] = True
            elif val > percents[3][j] and not flags[3]:
                xstats['uq'].append(x[i])
                flags[3] = True
            elif val > percents[4][j] and not flags[4]:
                xstats['98p'].append(x[i])
                flags[4] = True
        if not flags[4]:
            # this is what happens if it finds none of the above
            xstats['2p'].append(0.)
            xstats['lq'].append(0.)
            xstats['median'].append(0.)
            xstats['uq'].append(0.)
            xstats['98p'].append(0.)
    for k, v in xstats.items():
        xstats[k] = np.array(v)
    return xstats


def freq_spectrum(data, sampleRate, norm="forward", remove_dc_component=True):
    """
    Return the frequency spectrum of a data set. Combines negative and
    positive frequencies in the positive frequency range. Returns frequencies
    up until the Nyquist frequency f_N.

    Parameters
    ----------
    data : array_like, shape (m, ) or shape(n, m)
        The array of time signals where ``n`` is the number of variables and
        ``m`` is the number of time steps.
    sampleRate : int
        The signal sampling rate in Hertz.
    norm : str, optional
        Normalization of the returned spectrum. See
        https://numpy.org/doc/stable/reference/routines.fft.html#normalization
        for explanation. The default is "forward", which normalizes the
        frequency spectrum by 1/N.
    remove_dc_component : bool, optional
        If True, the DC component (f = 0) is not included in the returned
        spectrum ]0,f_N[. If False the returned spectrum covers
        [0, f_N[. The default is True.


    Returns
    -------
    frequency : ndarray, shape (p,)
        Frequencies where ``p`` is a power of 2 close to ``m``, in Hertz.
    amplitude : ndarray, shape (p, n)
        Amplitude at each frequency.

    Examples
    --------

    Create a sum of two sinusoids with the low frequency sinusoid having an
    amplitude of 2 and frequency of 5 Hz and the high frequency sinusoid having
    an amplitude of 1 and frequency of 50 Hz. Plot the frequency spectrum of
    the sum.

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import freq_spectrum

       sample_rate = 1000
       duration = 1.0
       time = np.linspace(0.0, duration, num=int(duration*sample_rate) + 1)

       low_freq = 2.0*np.sin(5.0*2.0*np.pi*time)  # 5 Hz * 2 pi rad / cycle
       high_freq = np.sin(50.0*2.0*np.pi*time)  # 50 Hz * 2 pi rad / cycle

       fig, ax = plt.subplots(layout='constrained')
       ax.plot(time, low_freq + high_freq)
       ax.set_xlim((0.0, 0.4))
       ax.set_xlabel('Time [s]')
       ax.set_ylabel('Amplitude')

    .. plot::
       :context: close-figs
       :include-source:

       freqs, amps = freq_spectrum(low_freq + high_freq, sample_rate)

       fig, ax = plt.subplots(layout='constrained')
       ax.plot(freqs, amps)
       ax.set_xlim((0.0, 100.0))
       ax.set_xlabel('Frequency [Hz]')
       ax.set_ylabel('Amplitude')

    """
    def nextpow2(i):
        '''Return the next power of 2 for the given number.'''
        n = 2
        while n < i:
            n *= 2
        return n

    time = 1./sampleRate  # sample time
    try:
        L = data.shape[1]  # length of data if (n, m)
    except IndexError:
        L = data.shape[0]  # length of data if (n,)
    # calculate the closest power of 2 for the length of the data
    n = nextpow2(L)

    Y = fft(data, n, norm="forward")
    f = fftfreq(n, d=time)
    # f = sampleRate/2.*linspace(0, 1, n)
    # print 'f =', f, f.shape, type(f)
    frequency = f[int(remove_dc_component):n//2]
    try:
        # multiply by 2 because we take half the vector
        amplitude = 2*abs(Y[:, int(remove_dc_component):n//2]).T
        # power = abs(Y[:, 1:n/2])**2
    except IndexError:
        amplitude = 2*abs(Y[int(remove_dc_component):n//2])
        # power = abs(Y[1:n/2])**2

    # Correct the dc component. It may not be multiplied by two,
    # the full spectrum [0,f_s[ (or equiv. ]-f_n, f_n[) only includes it once.
    if not remove_dc_component:
        amplitude[0] = amplitude[0] / 2

    return frequency, amplitude


def power_spectrum(data, sample_rate, remove_dc_component=False):
    """
    Return the power spectrum of a signal::

        S(f) = |X(f)|^2

    Notes
    -----
    - power_spectrum() performs zero-padding. Parseval's
      theorem is satisfied for the padded input signal. Provide input signals
      with 2^p samples to prevent zero-padding.
    - The power contributions of positive and negative frequencies are
      combined in the positive half spectrum so that the results satisfy
      Parseval's theoreom on the interval [0, f_N].
    - If the dc component is removed with remove_dc_component=True, the results
      do not satisfy Parseval's theorem.

    Parameters
    ----------
    data : ndarray, shape (m,) or shape(n,m)
        The array of time signals where n is the number of variables and m is
        the number of time steps.
    sample_rate : int
        The signal sampling rate in Hertz.
    remove_dc_component : bool, optional
        If True, the DC component (f = 0) is not included in the returned
        spectrum ]0,f_N[. If False the returned spectrum covers
        [0, f_N[. The default is True.


    Returns
    -------
    frequency : ndarray, shape (p,)
        The frequencies where p is a power of 2 close to m.
    power : ndarray, shape (p,n)
        The power at each frequency.

    Examples
    --------

    Create the power spectrum of a rect pulse and plot in time and frequency
    domain. Note how the power of frequencies f>0 is larger then f=0 because
    the positive frequencies inlclude the contribution of negative frequencies.
    As a result, the mean power in the displayed half spectrum equals the
    the mean power of the input signal.

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import power_spectrum

       # sampling parameters
       N = 64   # signal period
       f_s = 10  # sample rate
       T = N/f_s

       t = np.arange(0, T, 1/f_s)

       # rectangle test signal
       A = 3  # amplitude
       tau = 0.2*T # "on"-time

       x = np.zeros_like(t)
       x[0:int(tau*f_s)] = A

       # power spectrum
       freq, amp = power_spectrum(x, f_s)

       # check Parseval's theorem
       energy_time = np.mean(np.abs(x)**2)
       energy_freq = np.sum(amp)

       print(f"Mean power in time domain: {energy_time:.6f}")
       print(f"Mean power in frequency domain: {energy_freq:.6f}")

       # plot
       fig, ax = plt.subplots(2, 1, layout="constrained")
       ax[0].stem(t, x)
       ax[0].set_xlabel("$t$ in s")
       ax[0].set_ylabel("$x(t)$")
       ax[1].stem(freq,amp)
       ax[1].set_xlabel("$f$ in Hz")
       ax[1].set_ylabel("$|X(f)|^2$")
       plt.suptitle(f"Sample rate: {f_s} Hz, Signal period: {T} s")

    """
    # call freq_spectrum with orthonormal normalization (i.e. 1/sqrt(N)) to
    # ensure that Parseval's theorem is satisfied.
    frequency, amplitude = freq_spectrum(
        data, sample_rate, norm="ortho",
        remove_dc_component=remove_dc_component)

    # Power is the square of the amplitude.
    power = amplitude**2

    # Division by two is necessary to compensate doubelled amplitude of
    # freq_spectrum for f>0. (Freq_spectrum combines
    # the amplitude of the positive and negative frequencies).
    if not remove_dc_component:
        power[1:] = power[1:] / 2

    return frequency, power


def cumulative_power_spectrum(data, sample_rate, relative=True,
                              remove_dc_component=False):
    r"""
    Return the cumulative power spectrum of a signal::

       S(f) = \sum_{k=0}^f |X(k)|^2

    Notes
    -----

    - ``cumulative_power_spectrum()`` performs zero-padding. Parseval's theorem
      is satisfied for the padded input signal. Provide input signals with 2^p
      samples to prevent zero-padding.
    - The power contributions of positive and negative frequencies are
      combined in the positive half spectrum so that the results satisfy
      Parseval's theoreom on the interval ``[0, f_N]``.
    - If the dc component is removed with ``remove_dc_component=True``, the
      results do not satisfy Parseval's theorem.

    Parameters
    ----------
    data : ndarray, shape (m,) or shape(n,m)
        The array of time signals where n is the number of variables and m is
        the number of time steps.
    sample_rate : int
        The signal sampling rate in Hertz.
    relative : bool, optional
        If True, the returned amplitued is expressed relative to the total
        power. The default is True.
    remove_dc_component : bool, optional
        If True, the DC component (f = 0) is not included in the returned
        spectrum ``]0,f_N[``. If False the returned spectrum covers ``[0,
        f_N[``.  The default is False.

    Returns
    -------
    frequency : ndarray, shape (p,)
        The frequencies where p is a power of 2 close to m.
    cumulative_power : ndarray, shape (p,n)
        The cumulative power up to each frequency.

    Examples
    --------

    Create the cumulative power spectrum of a rect pulse and plot in time and
    frequency domain.

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import cumulative_power_spectrum

       # sampling parameters
       N = 64  # signal period
       f_s = 10  # sample rate
       T = N/f_s

       t = np.arange(0, T, 1/f_s)

       # rect test signal
       A = 3  # amplitude
       tau = 0.2*T  # "on"-time

       x = np.zeros_like(t)
       x[0:int(tau*f_s)] = A

       # power spectrum
       freq, amp = cumulative_power_spectrum(x, f_s)

       # plot
       fig, ax = plt.subplots(2,1, layout="constrained")
       ax[0].stem(t, x, )
       ax[0].set_xlabel("$t$ in s")
       ax[0].set_ylabel("$x(t)$")
       ax[1].stem(freq,amp)
       ax[1].set_xlabel("$f$ in Hz")
       ax[1].set_ylabel("cumulative avg. power")
       plt.suptitle((f"Sample rate: {f_s} Hz, Signal period: {T} s,"
                    " relative=True"))

    """
    frequency, power = power_spectrum(data, sample_rate,
                                      remove_dc_component=remove_dc_component)

    cumulative_power = np.cumsum(power)

    # if requested, normalize to the total power.
    if relative:
        cumulative_power = cumulative_power / cumulative_power[-1]

    return frequency, cumulative_power


def butterworth(data, cutoff, samplerate, order=2, axis=-1, btype='lowpass',
                **kwargs):
    """Returns the data filtered by a two-pass (forward/backward) Butterworth
    filter.

    Parameters
    ----------
    data : array_like, shape(n, ) or shape(n, m)
        The data to filter. Only handles 1D and 2D arrays.
    cutoff : float
        The filter cutoff frequency in Hertz.
    samplerate : float
        The sample rate of the data in Hertz. Sample rate must be constant.
    order : int
        The order of the Butterworth filter.
    axis : int
        The axis to filter along.
    btype : {'lowpass'|'highpass'}
        The type of filter. Default is 'lowpass'.
    kwargs
        Any extra arguments to get passed to scipy.signal.sosfiltfilt.

    Returns
    -------
    filtered_data : ndarray
        The low pass filtered version of data.

    Notes
    -----
    The provided cutoff frequency is corrected by a multiplicative factor to
    ensure the double pass filter cutoff frequency matches that of a single
    pass filter, see [Winter2009]_.

    References
    ----------
    .. [Winter2009] David A. Winter (2009) Biomechanics and motor control of
       human movement. 4th edition. Hoboken: Wiley.

    Examples
    --------

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import butterworth, freq_spectrum

       sample_rate = 1000  # Hz
       duration = 10.0  # seconds
       time = np.linspace(0.0, duration, num=int(sample_rate*duration) + 1)
       white_noise = np.random.normal(0.0, 1.0, size=len(time))
       cutoff = 200  # Hz
       order = 4
       filtered = butterworth(white_noise, cutoff, sample_rate, order=order)

       freq, amp = freq_spectrum(white_noise, sample_rate)
       freq_filt, amp_filt = freq_spectrum(filtered, sample_rate)

       fig, ax = plt.subplots(layout='constrained')
       ax.plot(freq, amp, label='Unfiltered')
       ax.plot(freq_filt, amp_filt, alpha=0.75, label='Filtered')
       ax.axvline(cutoff, color='black')
       ax.set_ylabel('Amplitude of White Noise with STD=1')
       ax.set_xlabel('Frequency [Hz]')
       msg = 'Sample rate: {} Hz, Cutoff: {} Hz, Order: {}'
       ax.set_title(msg.format(sample_rate, cutoff, order))
       ax.legend()

    """
    if len(data.shape) > 2:
        raise ValueError('This function only works with 1D or 2D arrays.')

    # NOTE : For details, start in this issue:
    # https://github.com/moorepants/DynamicistToolKit/issues/37
    # For a digital Butterworth filter, the cutoff frequency in rad/s is:
    # wc = tan(pi/fs*fc) where:
    # fc : cutoff frequency in Hz
    # fs : sample rate in Hz
    # wc : cutoff frequency in rad/s

    correction_factor = (np.sqrt(2.0) - 1.0)**(1.0/(2.0*order))
    cutoff_radps = np.tan(np.pi*cutoff/samplerate)
    if btype == 'highpass':
        cutoff_corrected_hz = samplerate/np.pi*np.arctan(
            cutoff_radps*correction_factor)
    elif btype == 'lowpass':
        cutoff_corrected_hz = samplerate/np.pi*np.arctan(
            cutoff_radps/correction_factor)
    else:
        raise ValueError("btype='{}' not supported.".format(btype))

    # Wn is the ratio of the corrected cutoff frequency to the Nyquist
    # frequency.
    nyquist_frequency = samplerate/2
    Wn = cutoff_corrected_hz / nyquist_frequency

    sos = butter(order, Wn, btype=btype, output='sos')

    return sosfiltfilt(sos, data, axis=axis, **kwargs)


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

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.process import subtract_mean
    >>> t = np.linspace(0.0, 2*np.pi, num=11)
    >>> y1 = np.sin(t)
    >>> y2 = np.sin(t) + 0.3
    >>> print(np.allclose(y1, subtract_mean(y2)))
    True
    >>> y2[2:5] = np.nan
    >>> subtract_mean(y2, hasNans=True)
    array([ 0.31123729,  0.89902254,         nan,         nan,         nan,
            0.31123729, -0.27654797, -0.63981923, -0.63981923, -0.27654797,
            0.31123729])

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

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.process import normalize
    >>> t = np.linspace(0.0, 2*np.pi, num=11)
    >>> y = 5.0*np.sin(t)
    >>> float(np.max(normalize(y)))
    1.0

    '''
    # TODO : This could be a try/except statement instead of an optional
    # argument.
    if hasNans:
        normSig = sig / np.nanmax(sig)
    else:
        normSig = sig / np.max(sig)

    return normSig


def derivative(x, y, method='forward', padding=None):
    """Returns the derivative of y with respect to x.

    Parameters
    ----------
    x : ndarray, shape(n,)
        The monotonically increasing independent variable.
    y : ndarray, shape(n,) or shape(n, m)
        The dependent variable(s).
    method : string, optional
        'forward'
            Use the forward difference method.
        'backward'
            Use the backward difference method.
        'central'
            Use the central difference method.
        'combination'
            This is equivalent to ``method='central', padding='second
            order'`` and is in place for backwards compatibility. Selecting
            this method will ignore and user supplied padding settings.
    padding : None, float, 'adjacent' or 'second order', optional
        The default, None, will result in the derivative vector being n-a in
        length where a=1 for forward and backward and a=2 for central. If
        you provide a float this value will be used to pad the result so
        that len(dydx) == n. If 'adjacent' is used, the nearest neighbor
        will be used for padding. If 'second order' is chosen second order
        foward and backward difference are used to pad the end points.

    Returns
    -------
    dydx : ndarray, shape(n,) or shape(n-1,)
        for combination else shape(n-1,)

    Examples
    --------

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       import matplotlib.pyplot as plt
       from dtk.process import derivative

       x = np.linspace(-10.0, 10.0, num=201)
       y = x**2

       fig, axes = plt.subplots(2, 1, layout='constrained')

       axes[0].plot(x, y)
       axes[1].plot(x, derivative(x, y, method='combination'))

    """
    x = np.asarray(x)
    y = np.asarray(y)

    if method == 'combination':
        method = 'central'
        padding = 'second order'

    if len(x.shape) > 1:
        raise ValueError('x must be have shape(n,).')

    if len(y.shape) > 2:
        raise ValueError('y can at most have two dimensions.')

    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same first dimension.')

    if method == 'forward' or method == 'backward':

        if x.shape[0] < 2:
            raise ValueError('x must have a length of at least 2.')

        if len(y.shape) == 1:
            deriv = np.diff(y) / np.diff(x)
        else:
            deriv = (np.diff(y.T) / np.diff(x)).T

    elif method == 'central':

        if x.shape[0] < 3:
            raise ValueError('x must have a length of at least 3.')

        if len(y.shape) == 1:
            deriv = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        else:
            deriv = ((y[2:] - y[:-2]).T / (x[2:] - x[:-2])).T

    else:

        msg = ("There is no {} method here! Try 'forward', 'backward', "
               "'central', or 'combination'.").format(method)
        raise NotImplementedError(msg)

    if padding is None:

        dydx = deriv

    else:

        dydx = np.zeros_like(y)

        if padding == 'adjacent':

            dydx[0] = deriv[0]
            dydx[-1] = deriv[-1]

        elif padding == 'second order':

            dydx[0] = ((-3.0*y[0] + 4.0*y[1] - y[2]) / 2.0 / (x[1] - x[0]))
            dydx[-1] = ((3.0*y[-1] - 4.0*y[-2] + y[-3]) / 2.0 /
                        (x[-1] - x[-2]))

        else:

            dydx[0] = padding
            dydx[-1] = padding

        if method == 'forward':
            dydx[:-1] = deriv
        elif method == 'backward':
            dydx[1:] = deriv
        elif method == 'central':
            dydx[1:-1] = deriv

    return dydx


def time_vector(num_samples, sample_rate, start_time=0.0):
    '''Returns a time vector starting at zero.

    Parameters
    ----------
    num_samples : int
        Total number of samples.
    sample_rate : float
        Sample rate of the signal in hertz.
    start_time : float, optional, default=0.0
        The start time of the time series.

    Returns
    -------
    time : ndarray, shape(numSamples,)
        Time vector starting at zero.

    Examples
    --------

    >>> from dtk.process import time_vector
    >>> time_vector(101, 100)
    array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
           0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
           0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
           0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
           0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
           0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
           0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
           0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
           0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
           0.99, 1.  ])

    '''

    ns = num_samples
    sr = float(sample_rate)

    return np.linspace(start_time, (ns - 1) / sr + start_time, num=ns)
