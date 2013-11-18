#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import itertools

# external libraries
import numpy as np
import matplotlib.pyplot as plt

def plot_phasor(eigenvalues, eigenvectors, components=None, compNames=None, show=False):
    """Returns a phasor plot of the given eigenvalues and eigenvectors.

    Parameters
    ----------
    eigenvalues : array_like, shape(n)
        The eigenvalues.
    eigenvectors : array_like, shape(n,n)
        The eigenvectors where each column corresponds to the eigenvalues.
    components : array_like, optional
        The indices of the eigenvector components to plot.
    show : boolean, optional, {True | *False*}
        If true the plots will be displayed.

    Returns
    -------
    figs : list
        A list of matplotlib figures.

    Notes
    -----
    Plots are not produced for zero eigenvalues.

    """

    figs = []
    lw = range(1, len(components) + 1)
    lw.reverse()
    for i, eVal in enumerate(eigenvalues):
        figs.append(plt.figure())
        ax = figs[-1].add_subplot(1, 1, 1, polar=True)
        if components is None:
            eVec = eigenvectors[:, i]
        else:
            eVec = eigenvectors[components, i]
        maxCom = abs(eVec).max()
        for j, component in enumerate(eVec):
            radius = abs(component) / maxCom
            theta = np.angle(component)
            ax.plot([0, theta], [0, radius], lw=lw[j])
        ax.set_rmax(1.0)
        ax.set_title('Eigenvalue: %1.3f$\pm$%1.3fj' % (eVal.real, eVal.imag))
        if compNames is not None:
            ax.legend(compNames)

    if show is True:
        for fig in figs:
            fig.show()

    return figs

def sort_modes(evals, evecs):
    """Sort a series of eigenvalues and eigenvectors into modes.

    Parameters
    ----------
    evals : ndarray, shape (n, m)
        eigenvalues
    evecs : ndarray, shape (n, m, m)
        eigenvectors

    """
    evalsorg = np.zeros_like(evals)
    evecsorg = np.zeros_like(evecs)
    # set the first row to be the same
    evalsorg[0] = evals[0]
    evecsorg[0] = evecs[0]
    # for each speed
    for i, speed in enumerate(evals):
        if i == evals.shape[0] - 1:
            break
        # for each current eigenvalue
        used = []
        for j, e in enumerate(speed):
            x, y = np.real(evalsorg[i, j]), np.imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = np.zeros(evals.shape[1])
            for k, eignext in enumerate(evals[i + 1]):
                xn, yn = np.real(eignext), np.imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = np.abs(((xn - x)**2 + (yn - y)**2)**0.5)
            if np.argmin(dist) in used:
                # set the already used indice higher
                dist[np.argmin(dist)] = np.max(dist) + 1.
            else:
                pass
            evalsorg[i + 1, j] = evals[i + 1, np.argmin(dist)]
            evecsorg[i + 1, :, j] = evecs[i + 1, :, np.argmin(dist)]
            # keep track of the indices we've used
            used.append(np.argmin(dist))
    return evalsorg, evecsorg

def eig_of_series(matrices):
    """Returns the eigenvalues and eigenvectors for a series of matrices.

    Parameters
    ----------
    matrices : array_like, shape(n,m,m)
        A series of square matrices.

    Returns
    -------
    eigenvalues : ndarray, shape(n,m)
        The eigenvalues of the matrices.
    eigenvectors : ndarray, shape(n,m,m)
        The eigenvectors of the matrices.

    """

    s = matrices.shape

    eigenvalues = np.zeros((s[0], s[1]), dtype=np.complex)
    eigenvectors = np.zeros(s, dtype=np.complex)

    for i, A in enumerate(matrices):
        eVal, eVec = np.linalg.eig(matrices[i])
        eigenvalues[i] = eVal
        eigenvectors[i] = eVec

    return eigenvalues, eigenvectors

def plot_root_locus(parvalues, eigenvalues, skipZeros=False,
        fig=None, parName=None, parUnits=None, **kwargs):
    """Returns a root locus plot of a series of eigenvalues with respect to a
    series of values.

    Parameters
    ----------
    parvalues : array_like, shape(n,)
        The parameter values corresponding to each eigenvalue.
    eigenvalues : array_like, shape(n,m)
        The m eigenvalues for each parameter value.
    skipZeros : boolean, optional, default = False
        If true any eigenvalues close to zero will not be plotted.
    fig : matplotlib.Figure, optional, default = None
        Pass in a figure to plot on.
    parName : string, optional
        Specify the name or abbreviation of the parameter name.
    parUnits : string, optional
        Specify the units of the parameter.
    **kwargs : varies
        Any option keyword argments for a matplotlib scatter plot.

    Returns
    -------
    fig : matplotlib.Figure

    """

    if fig is None:
        fig = plt.figure()
        needsBar = True
    else:
        needsBar = False

    ax = fig.add_subplot(1, 1, 1, aspect='equal')

    default = {'s': 20,
               'c': parvalues,
               'cmap': plt.cm.gist_rainbow,
               'edgecolors': 'none'}
    for k, v in default.items():
        if k not in kwargs.keys():
            kwargs[k] = v

    x = eigenvalues.real
    y = eigenvalues.imag

    if skipZeros is True:
        for i in range(x.shape[1]):
            if (abs(x[:, i] - np.zeros_like(x[:, i])) > 1e-8).any():
                scat = ax.scatter(x[:, i], y[:, i], **kwargs)
    else:
        for i in range(x.shape[1]):
            scat = ax.scatter(x[:, i], y[:, i], **kwargs)

    if needsBar is True:
        cb = fig.colorbar(scat)
        if parName is not None and parUnits is not None:
            cb.set_label('{} {}'.format(parName, parUnits))

    ax.grid(b=True)
    ax.set_xlabel('Real [1/s]')
    ax.set_ylabel('Imaginary [1/s]')

    return fig

def plot_root_locus_components(parvalues, eigenvalues, parts='both',
        parName=None, parUnits=None, skipZeros=True, ax=None, **kwargs):
    """Returns a root locus plot of a series of eigenvalues with respect to a
    series of values.

    Parameters
    ----------
    parvalues : array_like, shape(n,)
        The parameter values corresponding to each eigenvalue.
    eigenvalues : array_like, shape(n,m)
        The m eigenvalues for each parameter value.
    parts : string, optional, {*'both'*|'real'|'imaginary'}
        Specify whether both the real and imaginary lines should be plotted or
        one or the other.
    parName : string, optional
        Specify the name or abbreviation of the parameter name.
    parUnits : string, optional
        Specify the units of the parameter.
    skipZeros : boolean, optional, {*True*|False}
        If true any eigenvalues close to zero will not be plotted.
    **kwargs : varies
        Any option keyword argments for the matplotlib plot function. This will
        be applied to all lines.

    Returns
    -------
    fig : matplotlib.Figure
        Nothing is returned if an axis is provided.

    """
    newAx = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        newAx = True

    colors = itertools.cycle(plt.rcParams['axes.color_cycle'])

    for i, ev in enumerate(eigenvalues.T):
        # don't plot the zero eigenvalues
        isZero = (abs(ev.real - np.zeros_like(ev.real)) < 1e-14).all()
        if isZero and skipZeros==True:
            pass
        else:
            color = next(colors)
            if parts == 'both' or parts == 'imaginary':
                if (abs(ev.imag - np.zeros_like(ev.imag)) > 1e-14).any():
                   ax.plot(parvalues, ev.imag, '--', color=color,
                       label='Imaginary')
            if parts == 'both' or parts == 'real':
                ax.plot(parvalues, ev.real, '-', color=color, label='Real')

    ax.grid(b=True)
    ax.set_ylabel('Eigenvalue Component [$s^{-1}$]')
    ax.legend()

    if parName is not None and parUnits is not None:
        ax.set_xlabel('{} {}'.format(parName, parUnits))

    if newAx is True:
        if parName is not None:
            plt.title('Root locus with respect to {}'.format(parName))
        return fig

class Bode(object):
    """A class for creating Bode plots and the associated data."""
    def __init__(self, frequency, *args, **kwargs):
        """Returns a Bode object for a set of systems.

        Parameters
        ----------
        frequency : ndarray, shape(n,)
            An array of frequencies at which to evaluate the system frequency
            reponse in radians per second. Use numpy.logspace to generate them.
        sys : dtk.control.StateSpace object
            One or more state space systems. If more than one system is
            provided, they must all have the same inputs and outputs.

        """

        self.frequency = frequency

        self.systems = []
        for system in args:
            self.systems.append(system)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.mag_phase()

    def mag_phase(self):
        """Computes the magnitude and phase for all the systems in the Bode
        object.

        """

        self.magnitudes = []
        self.phases = []
        for system in self.systems:
            m, p = self.mag_phase_system(system)
            self.magnitudes.append(m)
            self.phases.append(p)

    def plot(self, **kwargs):
        """Plots the Bode plots for all systems in the Bode object."""

        try:
            del self.figs
        except AttributeError:
            pass

        try:
            kwargs
        except NameError:
            kwargs = {}

        for i, system in enumerate(self.systems):
            try:
                kwargs['color'] = self.colors[i]
            except AttributeError:
                pass
            try:
                kwargs['linestyle'] = self.linestyles[i]
            except AttributeError:
                pass

            self.plot_system(system, self.magnitudes[i], self.phases[i],
                    **kwargs)

        #for f in self.figs:
            #leg = f.phaseAx.legend(loc=4)
            #plt.setp(leg.get_texts(), fontsize='6.0') #'xx-small')

    def show(self):
        for f in self.figs:
            f.show()

    def mag_phase_system(self, system):
        """Returns the magnitude and phase for a single system.

        Parameters
        ----------
        system : dtk.control.StateSpace
            A state space system.

        Returns
        -------
        magnitude : ndarray, shape(n, m, p)
            An array with the magnitude of the input-output transfer functions
            for each frequency.
        phase : ndarray, shape(n, m, p)
            An array with the phase of the in input-output transfer functions
            for each frequency in radians per second.

        Notes
        -----
        n : number of frequencies
        m : number of outputs
        p : number of inputs

        """

        A = system.A
        B = system.B
        C = system.C
        D = system.D

        I = np.eye(*A.shape)

        magnitude = np.zeros((len(self.frequency), system.numOutputs,
            system.numInputs))
        phase = np.zeros((len(self.frequency), system.numOutputs,
            system.numInputs))

        for i, w in enumerate(self.frequency):
            sImA_inv = np.linalg.inv(1j * w * I - A)
            G = np.dot(np.dot(C, sImA_inv), B) + D
            magnitude[i, :, :] = np.abs(G)
            phase[i, :, :] = np.angle(G)

        for i in range(system.numInputs):
            for o in range(system.numOutputs):
                phase[:, o, i] = np.unwrap(phase[:, o, i])

        return magnitude, phase

    def plot_system(self, system, magnitude, phase, decibel=True, degree=True,
            **kwargs):
        """Plots the Bode plots of a single system. If a system for this object
        has already been plotted, it will add new lines to the existing plots.

        Parameters
        ----------
        system : dtk.control.StateSpace
            The state space system.
        magnitude : ndarray, shape(n, m, p)
            An array with the magnitude of the input-output transfer functions
            for each frequency.
        phase : ndarray, shape(n, m, p)
            An array with the phase of the in input-output transfer functions
            for each frequency in radians per second.

        """

        # if plot hasn't been called yet, then make a new list
        try:
            self.figs
        except AttributeError:
            self.figs = []

        if degree is True:
            phase = np.rad2deg(phase)

        if decibel is True:
            magnitude = 20.0 * np.log10(magnitude)

        plotNum = 0
        for i in range(system.numInputs):
            for o in range(system.numOutputs):

                if len(self.figs) < (system.numInputs * system.numOutputs):
                    fig = plt.figure()

                    # These where here but seem to be messing things up.
                    #fig.yprops = dict(rotation=90,
                                  #horizontalalignment='right',
                                  #verticalalignment='center',
                                  #x=-0.01)
                    fig.yprops = {}

                    fig.axprops = {}

                    fig.suptitle('Input: {}, Output: {}'.format(system.inputNames[i],
                        system.outputNames[o]))

                    fig.magAx = fig.add_subplot(2, 1, 1, **fig.axprops)
                    fig.phaseAx = fig.add_subplot(2, 1, 2, **fig.axprops)

                    fig.magAx.set_ylabel('Magnitude [dB]', **fig.yprops)
                    fig.phaseAx.set_ylabel('Phase [deg]', **fig.yprops)
                    fig.phaseAx.set_xlabel('Frequency [rad/s]')
                    fig.axprops['sharex'] = fig.axprops['sharey'] = fig.magAx
                    fig.magAx.grid(b=True)
                    fig.phaseAx.grid(b=True)

                    plt.setp(fig.magAx.get_xticklabels(), visible=False)
                    plt.setp(fig.magAx.get_yticklabels(), visible=True)
                    plt.setp(fig.phaseAx.get_yticklabels(), visible=True)

                    self.figs.append(fig)
                else:
                    fig = self.figs[plotNum]

                # plot the lines

                fig.magAx.semilogx(self.frequency, magnitude[:, o, i],
                        label=system.name, **kwargs)

                fig.phaseAx.semilogx(self.frequency, phase[:, o, i],
                        label=system.name, **kwargs)

                plotNum += 1

class StateSpace(object):
    """A linear time invariant system described by its state space."""
    def __init__(self, A, B, C, D, **kwargs):
        """Returns a StateSpace object.

        Parameters
        ----------
        A : ndarray, shape(n,n)
            The state matrix.
        B : ndarray, shape(n,p)
            The input matrix.
        C : ndarray, shape(m,n)
            The output matrix.
        D : ndarray, shape(m,p)
            The feedforward matrix.
        name : string, optional
            A name of the system.
        stateNames : list, len(n), optional
            A list of names of each state in order corresponding to A.
        inputNames : list, len(p), optional
            A list of names of each input in order corresponding to B.
        outputNames : list, len(m), optional
            A list of names of each output in order corresponding to C.

        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        defaultAttributes = {'name': 'System',
         'stateNames': ['State' + str(i) for i in range(self.A.shape[0])],
         'inputNames': ['Input' + str(i) for i in range(self.B.shape[1])],
         'outputNames': ['Output' + str(i) for i in range(self.C.shape[0])],
         }

        self.numStates = A.shape[0]
        self.numInputs = B.shape[1]
        self.numOutputs = C.shape[0]

        for attr, default in defaultAttributes.items():
            try:
                setattr(self, attr, kwargs[attr])
            except KeyError:
                setattr(self, attr, default)

    def __str__(self):
        return "A {} system with {} states, {} inputs and {} outputs."\
                .format(self.name, len(self.stateNames), len(self.inputNames),
                        len(self.outputNames))

def bode(system, frequency, fig=None, label=None, title=None, color=None):
    """Creates a Bode plot of the given system.

    Parameters
    ----------
    system : tuple
        The system can be defined as a state space or the numerator and
        denominator of a transfer function. If defined in state space form it
        should include ndarrays for the state, input, output and feed-forward
        matrices, in that order. These should only be defined for a single
        input and single output. If in transfer function form the ndarrays for
        the numerator and denomonator coefficients must be provided.
    frequency : ndarray
        An array of frequencies at which to evaluate the system frequency
        reponse in radians per second.
    fig : matplotlib Figure instance, optional

    Returns
    -------
    magnitude : ndarray
        The magnitude in dB of the frequency response.
    phase : ndarray
        The phase in degrees of the frequency response.
    fig : matplotlib Figure instance
        The Bode plot.

    """
    if fig is None:
        fig = plt.figure()

    fig.yprops = dict(rotation=90,
                  horizontalalignment='right',
                  verticalalignment='center',
                  x=-0.01)

    fig.axprops = {}
    # axes [left, bottom, width, height]
    fig.ax1 = fig.add_axes([.125, .525, .825, .275], **fig.axprops)
    fig.ax2 = fig.add_axes([.125, .2, .825, .275], **fig.axprops)

    magnitude = np.zeros(len(frequency))
    phase = np.zeros(len(frequency))

    try:
        A, B, C, D = system
    except ValueError:
        num, den = system
        n = np.poly1d(num)
        d = np.poly1d(den)
        Gjw = n(1j * frequency) / d(1j * frequency)
        magnitude = 20. * np.log10(np.abs(Gjw))
        phase = 180. / np.pi * np.unwrap(np.arctan2(np.imag(Gjw), np.real(Gjw)))
    else:
        I = np.eye(A.shape[0])
        for i, f in enumerate(frequency):
            # this inverse is expensive, can this be reformed to be solved with
            # a faster method?
            sImA_inv = np.linalg.inv(1j * f * I - A)
            G = np.dot(np.dot(C, sImA_inv), B) + D
            magnitude[i] = 20. * np.log10(np.abs(G))
            phase[i] = np.angle(G)
        phase = 180. / np.pi * np.unwrap(phase)

    fig.ax1.semilogx(frequency, magnitude, label=label)

    if title:
        fig.ax1.set_title(title)

    fig.ax2.semilogx(frequency, phase, label=label)

    fig.axprops['sharex'] = fig.axprops['sharey'] = fig.ax1
    fig.ax1.grid(b=True)
    fig.ax2.grid(b=True)

    plt.setp(fig.ax1.get_xticklabels(), visible=False)
    plt.setp(fig.ax1.get_yticklabels(), visible=True)
    plt.setp(fig.ax2.get_yticklabels(), visible=True)
    fig.ax1.set_ylabel('Magnitude [dB]', **fig.yprops)
    fig.ax2.set_ylabel('Phase [deg]', **fig.yprops)
    fig.ax2.set_xlabel('Frequency [rad/s]')

    if label:
        fig.ax1.legend()

    if color:
        print color
        plt.setp(fig.ax1.lines, color=color)
        plt.setp(fig.ax2.lines, color=color)

    return magnitude, phase, fig
