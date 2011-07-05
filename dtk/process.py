#!/usr/bin/env python

import numpy as np
from numpy.fft import fft, fftfreq
from scipy.integrate import trapz, cumtrapz
from scipy.signal import butter, lfilter

def curve_area_stats(x, y):
    '''
    Return the box plot stats of a curve based on area.

    Parameters:
    -----------
    x : ndarray, shape (n,)
        The x values
    y : ndarray, shape (n,m)
        The y values
        n are the time steps
        m are the various curves

    Returns:
    --------
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

def freq_spectrum(Fs, Data):
    '''
    Return the frequency spectrum of a data set.

    Parameters:
    -----------

    Fs : int
        sampling frequency
    Data : ndarray, shape (n,m)
        the time history vector
        n is the number of variables
        m is the number of time steps

    Returns:
    --------

    freq : ndarray, shape (p,)
        the frequencies where p a power of 2 close to m
    amp : ndarray, shape (p,n)

    '''
    def nextpow2(i):
        '''
        Return the next power of 2 for the given number.

        '''
        n = 2
        while n < i: n *= 2
        return n

    T = 1./Fs # sample time
    #print 'T =', T
    try:
        L = Data.shape[1] # length of Data if (n, m)
    except:
        L = Data.shape[0] # length of Data if (n,)
    #print 'L =', L
    # calculate the closest power of 2 for the length of the data
    n = nextpow2(L)
    #print 'n =', n
    Y = fft(Data, n)/L # divide by L for scaling
    #print 'Y =', Y, Y.shape, type(Y)
    f = fftfreq(n, d=T)
    #f = Fs/2.*linspace(0, 1, n)
    #print 'f =', f, f.shape, type(f)
    freq = f[1:n/2]
    try:
        amp = 2*abs(Y[:, 1:n/2]).T # multiply by 2 because we take half the vector
        #power = abs(Y[:, 1:n/2])**2
    except:
        amp = 2*abs(Y[1:n/2])
        #power = abs(Y[1:n/2])**2
    return freq, amp

def butterworth(data, freq, sampRate, order=2, axis=-1):
    """
    Returns the Butterworth filtered data set.

    Parameters:
    -----------
    data : ndarray
    freq : float or int
        cutoff frequency in hertz
    sampRate : float or int
        sampling rate in hertz
    order : int
        the order of the Butterworth filter
    axis : int
        the axis to filter along

    Returns:
    --------
    filteredData : ndarray
        filtered version of data

    This does a forward and backward Butterworth filter and averages the two.

    """
    nDim = len(data.shape)
    dataSlice = '['
    for dim in range(nDim):
        if dim == axis or (np.sign(axis) == -1 and dim == nDim + axis):
            dataSlice = dataSlice + '::-1, '
        else:
            dataSlice = dataSlice + ':, '
    dataSlice = dataSlice[:-2] + '].copy()'

    b, a = butter(order, float(freq) / float(sampRate) / 2.)
    forwardFilter = lfilter(b, a, data, axis=axis)
    reverseFilter = lfilter(b, a, eval('data' + dataSlice), axis=axis)
    return(forwardFilter + eval('reverseFilter' + dataSlice)) / 2.

def subtract_mean(sig):
    '''
    Subtracts the mean from a signal with nanmean.

    Parameters
    ----------
    sig : ndarray, shape(n,)

    Returns
    -------
    ndarray, shape(n,)
        sig minus the mean of sig

    '''
    return sig - nanmean(sig)

def normalize(sig):
    '''
    Normalizes the vector with respect to the maximum value.

    Parameters
    ----------
    sig : ndarray, shape(n,)

    Returns
    -------
    normSig : ndarray, shape(n,)
        The signal normalized with respect to the maximum value.

    '''
    return sig / np.nanmax(sig)

def derivative(x, y, method='forward'):
    '''
    Return the derivative of y with respect to x.

    Parameters:
    -----------
    x : ndarray, shape(n,)
    y : ndarray, shape(n,)
    method : string
        'forward' : forward difference
        'central' : central difference
        'backward' : backward difference
        'combination' : forward on the first point, backward on the last and
        central on the rest

    Returns:
    --------
    dydx : ndarray, shape(n,) or shape(n-1,)
        for combination else shape(n-1,)

    The combo method doesn't work for matrices yet.

    '''
    if method == 'forward':
        return np.diff(y)/np.diff(x)
    elif method == 'combination':
        dxdy = np.zeros_like(y)
        for i, yi in enumerate(y[:]):
            if i == 0:
                dxdy[i] = (-3*y[0] + 4*y[1] - y[2])/2/(x[1]-x[0])
            elif i == len(y) - 1:
                dxdy[-1] = (3*y[-1] - 4*y[-2] + y[-3])/2/(x[-1] - x[-2])
            else:
                dxdy[i] = (y[i + 1] - y[i - 1])/2/(x[i] - x[i - 1])
        return dxdy
    else:
        raise NotImplementedError('There is no %s method here! Try Again' %
            method)

def time_vector(numSamples, sampleRate):
    '''Returns a time vector starting at zero.

    Parameters
    ----------
    numSamples : int or float
        Total number of samples.
    sampleRate : int or float
        Sample rate.

    Returns
    -------
    time : ndarray, shape(numSamples,)
        Time vector starting at zero.

    '''
    ns = float(numSamples)
    sr = float(sampleRate)
    return np.linspace(0., (ns - 1.) / sr, num=ns)
