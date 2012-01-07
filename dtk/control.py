import numpy as np
import matplotlib.pyplot as plt

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
