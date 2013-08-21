#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external
import numpy as np
import matplotlib.pyplot as plt

# local
from dtk import process


class SimpleControlSolver(object):
    """This assumes a simple linear control structure at each time instance
    in a step.

    m(t) = K(t) [sc(t) - s(t)] = mc(t) - K(t) s(t)

    This class solves for the time dependent gains and the "commanded"
    controls using a simple linear least squares.

    """

    def __init__(self, data_panel):
        """Initializes the solver.

        Parameters
        ==========
        data_panel : pandas.Panel, shape(m, n, u)
            A panel in which each item is a data frame of the time series of
            various measured sensors with time as the index. This should not
            have any missing values.

        """
        self.data_panel = data_panel

    def solve(self, sensors, controls):
        """Returns the estimated gains and sensor limit cycles along with
        their variance.

        Parameters
        ==========
        sensors : sequence of strings
            A sequence of p strings which match column names in the data
            panel for the sensors.
        controls : sequence of strings
            A sequence of q strings which match column names in the data
            panel for the controls.

        Returns
        =======
        gain_matrix : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        sensor_limit_cycle : ndarray, shape(n, q)
            The estimated commanded sensor values.

        """
        self.sensors = sensors
        self.controls = controls

        self.lengths()

        A, b = self.form_a_b()

        x, variance, covariance = self.least_squares(A, b)

        gains, controls, sensors = self.deconstruct_solution(x)

        # TODO : output variances
        #gain_variance : ndarray, shape(n, q, p)
            #The variance in the estimated gain given the quality of fit.
        #sensor_limit_cycle_variance : ndarray(n, q)
            #The variance in the estimated sensor limit cycle with respect to
            #the quality of fit.

        return gains, sensors

    def deconstruct_solution(self, x):
        """Returns the gain matrices, command controls, and commanded
        sensors for each time step.

        m(t) = K(t) [sc(t) - s(t)] = mc(t) - K(t) s(t)

        Returns
        =======
        gain_matrices : ndarray,  shape(n, q, p)
            The gain matrices at each time step, K(t).
        control_star_vectors : ndarray, shape(n, q)
            The commanded controls, mc(t).
        sensor_star_vectors : shape(n, p)
            The commanded sensors, sc(t).

        """

        gain_matrices = np.zeros((self.n, self.q, self.p))
        control_star_vectors = np.zeros((self.n, self.q))
        sensor_star_vectors = np.zeros((self.n, self.p))

        # TODO : I'm not dealing with the steps/cycles correctly here. The
        # sensor star vectors are different for every cycle.
        control_vectors = self.form_control_vectors()[0]
        sensor_vectors = self.form_sensor_vectors()[0]

        for i in range(self.n):
            k_start = i * self.q * (self.p + 1)
            k_end = self.q * ((i + 1) * self.p + i)
            m_end = (i + 1) * self.q * (self.p + 1)
            gain_matrices[i] = x[k_start:k_end].reshape(self.q, self.p)
            control_star_vectors[i] = x[k_end:m_end]
            # m = m* - Ks > m = Ks* - Ks > m = K(s* - s) > K^-1 m + s = s*
            # TODO : This isn't working
            #b = control_vectors[i] + np.dot(gain_matrices[i], sensor_vectors[i])
            #sensor_star_vectors[i] = np.linalg.solve(gain_matrices[i], b)
        sensor_star_vectors = None

        # TODO : deconstruct the covariance matrices

        return gain_matrices, control_star_vectors, sensor_star_vectors

    def least_squares(self, A, b):
        """Returns the solution to the linear least squares and the
        covariance matrix of the solution.

        Parameters
        ==========
        A : array_like, shape(n, m)
            The coefficient matrix of Ax = b.
        b : array_like, shape(n,)
            The right hand side of Ax = b.

        Returns
        =======
        x : ndarray, shape(m,)
            The best fit solution.
        variance : float
            The variance of the fit.
        covariance : ndarray, shape(m, m)
            The covariance of the solution.

        """

        num_equations, num_unknowns = A.shape
        if num_equations < num_unknowns:
            raise Exception('Please add some walking cycles. There is ' +
                'not enough data to solve for the number of unknowns.')

        x, sum_of_residuals, rank, s = np.linalg.lstsq(A, b)

        variance, covariance = process.least_squares_variance(A,
                                                              sum_of_residuals)

        return x, variance, covariance

    def lengths(self):
        """Returns the number of sensors, controls, steps cycles, and time
        steps.

        Returns
        =======
        n : integer
            The number of time steps.
        m : integer
            The number of step cycles.
        p : integer
            The number of sensors.
        q : integer
            The number of controls.

        """
        self.n = self.data_panel.shape[1]
        self.m = self.data_panel.shape[0]
        self.p = len(self.sensors)
        self.q = len(self.controls)

        return self.n, self.m, self.p, self.q

    def form_sensor_vectors(self):
        """Returns an array of sensor vectors for each cycle and each time step.

        Returns
        =======
        sensor_vectors : ndarray, shape(m, n, p)
            The sensor vector form the i'th cycle and the j'th time step
            will look like [sensor_0, ..., sensor_(p-1)].
        """
        sensor_vectors = np.zeros((self.m, self.n, self.p))
        for i, (panel_name, data_frame) in enumerate(self.data_panel.iteritems()):
            for j, (index, values) in enumerate(data_frame[self.sensors].iterrows()):
                sensor_vectors[i, j] = values.values

        return sensor_vectors

    def form_control_vectors(self):
        """Returns an array of control vectors for each cycle and each time
        step.

        Returns
        =======
        control_vectors : ndarray, shape(m, n, q)
            The sensor vector form the i'th cycle and the j'th time step
            will look like [control_0, ..., control_(q-1)].

        """
        control_vectors = np.zeros((self.m, self.n, self.q))
        for i, (panel_name, data_frame) in enumerate(self.data_panel.iteritems()):
            for j, (index, values) in enumerate(data_frame[self.controls].iterrows()):
                control_vectors[i, j] = values.values

        return control_vectors

    def form_a_b(self):
        """Returns the A matrix and the b vector for the linear least
        squares fit.

        Returns
        =======
        A : ndarray, shape(n * q, n * q * (p + 1))
            The A matrix which is sparse and contains the sensor
            measurements and ones.
        b : ndarray, shape(n * q,)
            The b vector which constaints the measured controls.

        """
        control_vectors = self.form_control_vectors()

        b = np.zeros((self.m * self.n * self.q))

        b = np.array([])
        for cycle in control_vectors:
            for time_step in cycle:
                b = np.hstack((b, -time_step))

        sensor_vectors = self.form_sensor_vectors()

        A = np.zeros((self.m * self.n * self.q, self.n * self.q * (self.p + 1)))
        for i in range(self.m):
            Am = np.zeros((self.n * self.q, self.n * self.q * (self.p + 1)))
            for j in range(self.n):
                An = np.zeros((self.q, self.q * self.p))
                for row in range(self.q):
                    An[row, row * self.p:(row + 1) * self.p] = \
                        sensor_vectors[i, j]
                An = np.hstack((An, -np.eye(self.q)))
                num_rows, num_cols = An.shape
                Am[j * num_rows:(j + 1) * num_rows, j * num_cols:(j + 1) *
                    num_cols] = An
            A[i * self.n * self.q:i * self.n * self.q + self.n * self.q] = Am

        return A, b


def find_constant_speed(time, speed, plot=False):
    """Returns the indice at which the treadmill speed becomes constant and
    the time series when the treadmill speed is constant.

    Parameters
    ==========
    time : array_like, shape(n,)
        A monotonically increasing array.
    speed : array_like, shape(n,)
        A speed array, one sample for each time. Should ramp up and then
        stablize at a speed.
    plot : boolean, optional
        If true a plot will be displayed with the results.

    Returns
    =======
    indice : integer
        The indice at which the speed is consider constant thereafter.
    new_time : ndarray, shape(n-indice,)
        The new time array for the constant speed section.

    """

    sample_rate = 1.0 / (time[1] - time[0])

    filtered_speed = process.butterworth(speed, 3.0, sample_rate)

    acceleration = np.hstack( (0.0, np.diff(filtered_speed)) )

    noise_level = np.max(np.abs(acceleration[int(0.2 * len(acceleration)):-1]))

    reversed_acceleration = acceleration[::-1]

    indice = np.argmax(reversed_acceleration > noise_level)

    additional_samples = sample_rate * 0.65

    new_indice = indice - additional_samples

    if plot is True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(time, speed, '.', time, filtered_speed, 'g-')
        ax[0].plot(np.ones(2) * (time[len(time) - new_indice]),
            np.hstack( (np.max(speed), np.min(speed)) ))
        ax[1].plot(time, np.hstack((0.0, np.diff(filtered_speed))), '.')
        fig.show()

    return len(time) - (new_indice), time[len(time) - new_indice]


def gait_landmarks_from_grf(time, right_grfy, left_grfy,
                            threshold=1e-5, do_plot=False, min_time=None,
                            max_time=None):
    """
    Obtain gait landmarks (right and left foot strike & toe-off) from ground
    reaction force (GRF) time series data.

    Parameters
    ----------
    time : array_like, shape(n,)
        A monotonically increasing time array.
    right_grfy : array_like, shape(n,)
        Name of column in `mot_file` containing the y (vertical) component
        of GRF data for the right leg.
    left_grfy : str, shape(n,)
        Same as above, but for the left leg.
    threshold : float, optional
        Below this value, the force is considered to be zero (and the
        corresponding foot is not touching the ground).
    do_plot : bool, optional (default: False)
        Create plots of the detected gait landmarks on top of the vertical
        ground reaction forces.
    min_time : float, optional
        If set, only consider times greater than `min_time`.
    max_time : float, optional
        If set, only consider times greater than `max_time`.

    Returns
    -------
    right_foot_strikes : np.array
        All times at which right_grfy is non-zero and it was 0 at the
        preceding time index.
    left_foot_strikes : np.array
        Same as above, but for the left foot.
    right_toe_offs : np.array
        All times at which left_grfy is 0 and it was non-zero at the
        preceding time index.
    left_toe_offs : np.array
        Same as above, but for the left foot.

    Notes
    -----
    Source modifed from:

    https://github.com/fitze/epimysium/blob/master/epimysium/postprocessing.py

    """
    # Helper functions
    # ----------------
    def zero(number):
        return abs(number) < threshold

    def birth_times(ordinate):
        births = list()
        for i in index_range:
            # 'Skip' first value because we're going to peak back at previous
            # index.
            if zero(ordinate[i - 1]) and (not zero(ordinate[i])):
                births.append(time[i])
        return np.array(births)

    def death_times(ordinate):
        deaths = list()
        for i in index_range:
            if (not zero(ordinate[i - 1])) and zero(ordinate[i]):
                deaths.append(time[i])
        return np.array(deaths)

    def nearest_index(array, val):
            return np.abs(array - val).argmin()

    # Time range to consider.
    if max_time is None:
        max_idx = len(time)
    else:
        max_idx = nearest_index(time, max_time)

    if min_time is None:
        min_idx = 1
    else:
        min_idx = max(1, nearest_index(time, min_time))

    index_range = range(min_idx, max_idx)

    right_foot_strikes = birth_times(right_grfy)
    left_foot_strikes = birth_times(left_grfy)
    right_toe_offs = death_times(right_grfy)
    left_toe_offs = death_times(left_grfy)

    if do_plot:

        plt.figure(figsize=(10, 8))
        ones = np.array([1, 1])

        def myplot(index, label, ordinate, foot_strikes, toe_offs):
            ax = plt.subplot(2, 1, index)
            plt.plot(time[min_idx:max_idx], ordinate[min_idx:max_idx], 'k')
            plt.ylabel('vertical ground reaction force (N)')
            plt.title('%s (%i foot strikes, %i toe-offs)' % (
                label, len(foot_strikes), len(toe_offs)))
            #ax.set_ylim((-50.0, 50.0))

            for i, strike in enumerate(foot_strikes):
                if i == 0:
                    kwargs = {'label': 'foot strikes'}
                else:
                    kwargs = dict()
                plt.plot(strike * ones, ax.get_ylim(), 'r', **kwargs)

            for i, off in enumerate(toe_offs):
                if i == 0:
                    kwargs = {'label': 'toe-offs'}
                else:
                    kwargs = dict()
                plt.plot(off * ones, ax.get_ylim(), 'b', **kwargs)

        myplot(1, 'left foot', left_grfy, left_foot_strikes, left_toe_offs)
        plt.legend(loc='best')

        myplot(2, 'right foot', right_grfy, right_foot_strikes, right_toe_offs)

        plt.xlabel('time (s)')

    return right_foot_strikes, left_foot_strikes, right_toe_offs, left_toe_offs
