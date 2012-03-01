import numpy as np
import matplotlib.pyplot as plt

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
            if self.colors is not None:
                kwargs['color'] = self.colors[i]
            if self.linestyles is not None:
                kwargs['linestyle'] = self.linestyles[i]
            self.plot_system(system, self.magnitudes[i], self.phases[i],
                    **kwargs)

        for f in self.figs:
            leg = f.phaseAx.legend(loc=4)
            plt.setp(leg.get_texts(), fontsize='6.0') #'xx-small')

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

                    fig.yprops = dict(rotation=90,
                                  horizontalalignment='right',
                                  verticalalignment='center',
                                  x=-0.01)

                    fig.axprops = {}

                    fig.suptitle('Input: {}, Output: {}'.format(system.inputNames[i],
                        system.outputNames[o]))

                    # axes [left, bottom, width, height]
                    fig.magAx = fig.add_axes([.125, .525, .825, .275], **fig.axprops)
                    fig.phaseAx = fig.add_axes([.125, .2, .825, .275], **fig.axprops)

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

class System(object):
    """A placeholder class for a linear dynamic system."""
    pass

class StateSpace(System):
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
