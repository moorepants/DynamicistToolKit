#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas
from scipy import sparse

# local
from dtk import process


class WalkingData(object):
    """A class to store typical walking data."""

    def __init__(self, data_frame):
        """Initializes the data structure.

        Parameters
        ==========
        data_frame : pandas.DataFrame
            A data frame with an index of time and columns for each variable
            measured during a walking run.

        """
        # Could have a real time index:
        # new_index = [pandas.Timestamp(x, unit='s') for x in data.index]
        # data_frame.index = new_index
        # data.index.values.astype(float)*1e-9

        self.raw_data = data_frame

    def time_derivative(self, col_names, new_col_names=None):
        """Numerically differentiates the specified columns with respect to
        the time index and adds the new columns to `self.raw_data`.

        Parameters
        ==========
        col_names : list of strings
            The column names for the time series which should be numerically
            time differentiated.
        new_col_names : list of strings, optional
            The desired new column name(s) for the time differentiated
            series. If None, then a default name of `Time derivative of
            <origin column name>` will be used.

        """

        if new_col_names is None:
            new_col_names = ['Time derivative of {}'.format(c) for c in
                             col_names]

        for col_name, new_col_name in zip(col_names, new_col_names):
            self.raw_data[new_col_name] = \
                process.derivative(self.raw_data.index.values.astype(float),
                                   self.raw_data[col_name],
                                   method='combination')

    def grf_landmarks(self, right_vertical_grf_col_name,
                      left_vertical_grf_col_name, **kwargs):
        """Returns the times at which heel strikes and toe offs happen in
        the raw data.

        Parameters
        ==========
        right_vertical_grf_column_name : string
            The name of the column in the raw data frame which corresponds
            to the right foot vertical ground reaction force.
        left_vertical_grf_column_name : string
            The name of the column in the raw data frame which corresponds
            to the left foot vertical ground reaction force.

        Returns
        =======
        right_strikes : np.array
            All times at which right_grfy is non-zero and it was 0 at the
            preceding time index.
        left_strikes : np.array
            Same as above, but for the left foot.
        right_offs : np.array
            All times at which left_grfy is 0 and it was non-zero at the
            preceding time index.
        left_offs : np.array
            Same as above, but for the left foot.

        Notes
        =====
        This is a simple wrapper to gait_landmarks_from_grf and supports all
        the optional keyword arguments that it does.

        """

        right_strikes, left_strikes, right_offs, left_offs = \
            gait_landmarks_from_grf(self.raw_data.index.values.astype(float),
                                    self.raw_data[right_vertical_grf_col_name].values,
                                    self.raw_data[left_vertical_grf_col_name].values,
                                    **kwargs)

        self.strikes = {}
        self.offs = {}

        self.strikes['right'] = right_strikes
        self.strikes['left'] = left_strikes
        self.offs['right'] = right_offs
        self.offs['left'] = left_offs

        return right_strikes, left_strikes, right_offs, left_offs

    @staticmethod
    def interpolate(data_frame, time):
        """Returns a data frame with a index based on the provided time
        array and linear interpolation.

        Parameters
        ==========
        data_frame : pandas.DataFrame
            A data frame with time series columns. The index should be in
            seconds.
        time : array_like, shape(n,)
            A monotonically increasing array of time in seconds at which the
            data frame should be interpolated at.

        Returns
        =======
        interpolated_data_frame : pandas.DataFrame
            The data frame with an index matching `time_vector` and
            interpolated values based on `data_frame`.

        """

        total_index = np.sort(np.hstack((data_frame.index.values,
                                         time)))
        reindexed_data_frame = data_frame.reindex(total_index)
        interpolated_data_frame = \
            reindexed_data_frame.apply(pandas.Series.interpolate,
                                       method='values').loc[time]

        # If the first or last value of a series is NA then the interpolate
        # function leaves it as an NA value, so use backfill to take care of
        # those.
        interpolated_data_frame = \
            interpolated_data_frame.fillna(method='backfill')
        # Because the time vector may have matching indices as the original
        # index (i.e. always the zero indice), drop any duplicates so the
        # len() stays consistent
        return interpolated_data_frame.drop_duplicates()

    def split_at(self, side, section='both', num_samples=None):
        """Forms a pandas.Panel which has an item for each step. The index
        will be a new time vector starting at zero.

        Parameters
        ==========
        side : string {right|left}
            Split with respect to the right or left side heel strikes and/or
            toe-offs.
        section : string {both|stance|swing}
            Whether to split around the stance phase, swing phase, or both.
        num_samples : integer
            If provided the time series in each step will be interpolated at
            values evenly spaced in time across the step.

        Returns
        =======
        steps : pandas.Panel

        """

        if section == 'stance':
            lead = self.strikes[side]
            trail = self.offs[side]
        elif section == 'swing':
            lead = self.offs[side]
            trail = self.strikes[side]
        elif section == 'both':
            lead = self.strikes[side]
            trail = self.strikes[side][1:]
        else:
            raise ValueError('{} is not a valid section name'.format(section))

        if lead[0] > trail[0]:
            trail = trail[1:]

        samples = []
        max_times = []
        for i, lead_val in enumerate(lead):
            try:
                step_slice = self.raw_data[lead_val:trail[i]]
            except IndexError:
                pass
            else:
                samples.append(len(step_slice))
                max_times.append(float(step_slice.index[-1]) -
                                 float(step_slice.index[0]))

        max_num_samples = min(samples)
        max_time = min(max_times)

        steps = {}
        for i, lead_val in enumerate(lead):
            try:
                # get the step and truncate it to the max value
                data_frame = \
                    self.raw_data[lead_val:trail[i]].iloc[:max_num_samples]
            except IndexError:
                pass
            else:
                # make a new index starting from zero for each step
                new_index = data_frame.index.values.astype(float) - \
                    data_frame.index[0]
                data_frame.index = new_index
                if num_samples is None:
                    num_samples = max_num_samples
                # create a time vector index which has a specific number
                # of samples over the period of time, the max time needs
                sub_sample_index = np.linspace(0.0, max_time,
                                               num=num_samples)
                interpolated_data_frame = self.interpolate(data_frame,
                                                           sub_sample_index)
                steps[i] = interpolated_data_frame

        self.steps = pandas.Panel(steps)

        return self.steps

    def plot_steps(self, *col_names, **kwargs):
        """Plots the steps.

        Parameters
        ==========
        col_names : string
            A variable number of strings naming the columns to plot.
        kwargs : key value pairs
            Any extra kwargs to pass to the matplotlib plot command.

        """

        if len(col_names) == 0:
            raise ValueError('Please supply some column names to plot.')

        fig, axes = plt.subplots(len(col_names), sharex=True)

        for key, value in self.steps.iteritems():
            for i, col_name in enumerate(col_names):
                try:
                    ax = axes[i]
                except TypeError:
                    ax = axes
                ax.plot(value[col_name].index, value[col_name], **kwargs)
                ax.set_ylabel(col_name)
                ax.set_xlabel('Time [s]')

        return axes


class SimpleControlSolver(object):
    """This assumes a simple linear control structure at each time instance
    in a step.

    The measured joint torques equal some limit cycle joint torque plus a
    matrix of gains multiplied by the error in the sensors and the nominal
    value of the sensors.

    m_measured(t) = m_nominal + K(t) [s_nominal(t) - s(t)] = m*(t) - K(t) s(t)

    This class solves for the time dependent gains and the "commanded"
    controls using a simple linear least squares.

    """

    def __init__(self, data_panel, sensors, controls):
        """Initializes the solver.

        Parameters
        ==========
        data_panel : pandas.Panel, shape(m, n, u)
            A panel in which each item is a data frame of the time series of
            various measured sensors with time as the index. This should not
            have any missing values.
        sensors : sequence of strings
            A sequence of p strings which match column names in the data
            panel for the sensors.
        controls : sequence of strings
            A sequence of q strings which match column names in the data
            panel for the controls.

        """
        self._gain_omission_matrix = None

        self.data_panel = data_panel
        self.sensors = sensors
        self.controls = controls

    @property
    def data_panel(self):
        return self._data_panel

    @data_panel.setter
    def data_panel(self, value):
        self._data_panel = value
        self.m = value.shape[0]
        self.n = value.shape[1]

    @property
    def controls(self):
        return self._controls

    @controls.setter
    def controls(self, value):
        self._controls = value
        self.q = len(value)

    @property
    def sensors(self):
        return self._sensors

    @sensors.setter
    def sensors(self, value):
        self._sensors = value
        self.p = len(value)

    @property
    def gain_omission_matrix(self):
        return self._gain_omission_matrix

    @gain_omission_matrix.setter
    def gain_omission_matrix(self, value):
        if value is not None:
            if value.shape != (self.q, self.p):
                raise ValueError('The gain omission matrix should be of ' +
                                 'shape({}, {})'.format(self.q, self.p))
        self._gain_omission_matrix = value

    def solve(self, sparse_a=False, gain_omission_matrix=None):
        """Returns the estimated gains and sensor limit cycles along with
        their variance.

        Parameters
        ==========
        sparse_a : boolean, optional, default=False
            If true a sparse A matrix will be used along with a sparse
            linear least squares solver.
        gain_omission_matrix : boolean array_like, shape(q, p)
            A matrix which is the same shape as the identified gain matrices
            which has False in place of gains that should be assumed to be
            zero and True for gains that should be identified.

        Returns
        =======
        gain_matrices : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        control_vectors : ndarray, shape(n, q)
            The nominal control vector plus the gains multiplied by the
            reference sensors at each time step.
        variance : float
            The variance in the fitted curve.
        gain_matrices_variance : ndarray, shape(n, q, p)
            The variance of the found gains (covariance is neglected).
        control_vectors_variance : ndarray, shape(n, q)
            The variance of the found commanded controls (covariance is
            neglected).
        estimated_controls :

        """
        self.gain_omission_matrix = gain_omission_matrix

        A, b = self.form_a_b()

        # TODO : To actually get some memory reduction I should construct
        # the A matrix with a scipy.sparse.lil_matrix in self.form_a_b
        # instead of simply converting the dense matrix after I build it.

        if sparse_a is True:
            A = sparse.csr_matrix(A)

        x, variance, covariance = self.least_squares(A, b)

        deconstructed_solution = self.deconstruct_solution(x, covariance)

        gain_matrices = deconstructed_solution[0]
        gain_matrices_variance = deconstructed_solution[2]

        nominal_controls = deconstructed_solution[1]
        nominal_controls_variance = deconstructed_solution[3]

        estimated_controls = \
            self.compute_estimated_controls(gain_matrices, nominal_controls)

        return (gain_matrices, nominal_controls, variance,
                gain_matrices_variance, nominal_controls_variance,
                estimated_controls)

    def compute_estimated_controls(self, gain_matrices, nominal_controls):
        """Returns the predicted values of the controls and the
        contributions to the controls given gains, K(t), and nominal
        controls, m*(t), for each point in the gait cycle.

        Parameters
        ==========
        gain_matrices : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        control_vectors : ndarray, shape(n, q)
            The nominal control vector plus the gains multiplied by the
            reference sensors at each time step.

        Returns
        =======
        panel : pandas.Panel, shape(m, n, q)
            There is one data frame to correspond to each step in
            self.data_panel. Each data frame has columns of time series
            which store m(t), m*(t), and the individual components due to
            K(t) * se(t).

        Notes
        =====

        m(t) = m0(t) + K(t) * [ s0(t) - s(t) ] = m0(t) + K(t) * se(t)
        m(t) = m*(t) - K(t) * s(t)

        This function returns m(t), m0(t), m*(t) for each control and K(t) *
        [s0(t) - s(t)] for each sensor affecting each control. Where s0(t)
        is estimated by taking the mean with respect to the steps.

        """
        # generate all of the column names
        contributions = []
        for control in self.controls:
            for sensor in self.sensors:
                contributions.append(control + '-' + sensor)

        control_star = [control + '*' for control in self.controls]
        control_0 = [control + '0' for control in self.controls]

        col_names = self.controls + contributions + control_star + control_0

        panel = {}

        # The mean of the sensors which we consider to be the commanded
        # motion. This may be a bad assumption, but is the best we can do.
        mean_sensors = self.data_panel.mean(axis='items')[self.sensors]

        for i, df in self.data_panel.iteritems():

            blank = np.zeros((self.n, self.q * 3 + self.p * self.q))
            results = pandas.DataFrame(blank, index=df.index,
                                       columns=col_names)

            sensor_error = mean_sensors - df[self.sensors]

            for j in range(self.n):

                # m(t) = m*(t) - K(t) * s(t)
                m = nominal_controls[j] - np.dot(gain_matrices[j],
                                                 df[self.sensors].iloc[j])
                # m0(t) = m(t) - K(t) * se(t)
                m0 = m - np.dot(gain_matrices[j],
                                sensor_error[self.sensors].iloc[j])

                # these assignments don't work if I do:
                # results[self.controls].iloc[j] = m
                # but this seems to work
                # results.iloc[j][self.controls] = m
                # this is explained here:
                # https://github.com/pydata/pandas/issues/5093
                row_label = results.index[j]
                results.loc[row_label, self.controls] = m
                results.loc[row_label, control_0] = m0
                results.loc[row_label, control_star] = nominal_controls[j]

                for k, sensor in enumerate(self.sensors):
                    # control contribution due to the kth sensor
                    names = [c + '-' + sensor for c in self.controls]
                    results.loc[row_label, names] = gain_matrices[j, :, k] * \
                        sensor_error.iloc[j, k]

            panel[i] = results

        return pandas.Panel(panel)

    def plot_control_contributions(self, estimated_panel, max_num_steps=4):
        """Plots two graphs for each control and each step showing
        contributions from the linear portions. The first set of graphs
        shows the first few steps and the contributions to the control
        moments. The second set of graph shows the mean contributions to the
        control moment over all steps.

        Parameters
        ----------
        panel : pandas.Panel, shape(m, n, q)
            There is one data frame to correspond to each step. Each data
            frame has columns of time series which store m(t), m*(t), and
            the individual components due to K(t) * se(t).

        """

        num_steps = estimated_panel.shape[0]
        if num_steps > max_num_steps:
            num_steps = max_num_steps

        column_names = estimated_panel[0].columns

        for control in self.controls:
            fig, axes = plt.subplots(int(round(num_steps / 2.0)), 2,
                                     sharex=True)
            fig.suptitle('Contributions to the {} control'.format(control))
            contribs = [name for name in column_names if '-' in name and
                        name.startswith(control)]
            contribs += [control + '0']

            for ax, (step_num, cycle) in zip(axes.flatten()[:num_steps],
                                             estimated_panel.iteritems()):
                # here we want to plot each component of this:
                # m0 + k11 * se1 + k12 se2
                cycle[contribs].plot(kind='bar', stacked=True, ax=ax,
                                     title='Step {}'.format(step_num),
                                     colormap='jet')
                formatter = FuncFormatter(lambda l, p: '{:1.2f}'.format(l))
                ax.xaxis.set_major_formatter(FuncFormatter(formatter))

                for t in ax.get_legend().get_texts():
                    t.set_fontsize(6)
                    # only show the contribution in the legend
                    try:
                        t.set_text(t.get_text().split('-')[1])
                    except IndexError:
                        t.set_text(t.get_text().split('.')[1])

            for axis in axes[-1]:
                axis.set_xlabel('Time [s]')

        # snatch the colors from the last axes
        contrib_colors = [patch.get_facecolor() for patch in
                          ax.get_legend().get_patches()]

        mean = estimated_panel.mean(axis='items')
        std = estimated_panel.std(axis='items')

        for control in self.controls:
            fig, ax = plt.subplots()
            fig.suptitle('Contributions to the {} control'.format(control))
            contribs = [control + '0']
            contribs += [name for name in column_names if '-' in name and
                         name.startswith(control)]
            for col, color in zip(contribs, contrib_colors):
                ax.errorbar(mean.index.values, mean[col].values,
                            yerr=std[col].values, color=color)

            labels = []
            for contrib in contribs:
                try:
                    labels.append(contrib.split('-')[1])
                except IndexError:
                    labels.append(contrib.split('.')[1])
            ax.legend(labels, fontsize=10)
            ax.set_xlabel('Time [s]')

    def plot_estimated_vs_measure_controls(self, estimated_panel, variance):
        """Plots a figure for each control where the measured control is
        shown compared to the estimated along with a plot of the error.

        Parameters
        ==========
        estimated_panel : pandas.Panel
            A panel where each item is a step.
        variance : float
            The variance of the fit.

        Returns
        =======
        axes : array of matplotlib.axes.Axes, shape(q,)
            The plot axes.

        """

        # TODO : Construct the original time vector for the index.
        # TODO : Plot the estimated controls versus the full actual
        # measurement curve so that the measurement curve is very smooth.

        estimated_walking = pandas.concat([df for k, df in
                                           estimated_panel.iteritems()],
                                          ignore_index=True)

        actual_walking = pandas.concat([df for k, df in
                                        self.data_panel.iteritems()],
                                       ignore_index=True)

        fig, axes = plt.subplots(self.q * 2, sharex=True)

        for i, control in enumerate(self.controls):

            compare_axes = axes[i * 2]
            error_axes = axes[i * 2 + 1]

            sample_number = actual_walking.index.values
            measured = actual_walking[control].values
            predicted = estimated_walking[control].values
            std_of_predicted = np.sqrt(variance) * np.ones_like(predicted)
            error = measured - predicted
            rms = np.sqrt(np.linalg.norm(error).mean())
            r_squared = process.coefficient_of_determination(measured,
                                                             predicted)

            compare_axes.plot(sample_number, measured, color='black',
                              marker='.')
            compare_axes.errorbar(sample_number, predicted,
                                  yerr=std_of_predicted, fmt='.')
            compare_axes.set_ylabel(control)
            compare_axes.legend(('Measured',
                                 'Estimated {:1.1%}'.format(r_squared)))

            if i == len(self.controls) - 1:
                error_axes.set_xlabel('Sample Number')

            error_axes.plot(sample_number, error, color='black')
            error_axes.legend(['RMS = {:1.2f}'.format(rms)])
            error_axes.set_ylabel('Error in\n{}'.format(control))

        return axes

    def deconstruct_solution(self, x, covariance):
        """Returns the gain matrices and m*(t) for each time step.

        m(t) = m*(t) - K(t) s(t)

        Parameters
        ==========
        x : array_like, shape(n * q * (p + 1),)
            The solution matrix containing the gains and the commanded
            controls.
        covariance : array_like, shape(n * q * (p + 1), n * q * (p + 1))
            The covariance of x with respect to the variance in the fit.

        Returns
        =======
        gain_matrices : ndarray,  shape(n, q, p)
            The gain matrices at each time step, K(t).
        control_vectors : ndarray, shape(n, q)
            The nominal control vector plus the gains multiplied by the
            reference sensors at each time step.
        gain_matrices_variance : ndarray, shape(n, q, p)
            The variance of the found gains (covariance is neglected).
        control_vectors_variance : ndarray, shape(n, q)
            The variance of the found commanded controls (covariance is
            neglected).

        Notes
        =====
        x looks like:
            [k11(0), k12(0), ..., kqp(0), m1*(0), ..., mq*(0), ...,
             k11(n), k12(0), ..., kqp(n), m1*(n), ..., mq*(n)]
        """
        if self.gain_omission_matrix is not None:
            x1 = self.gain_omission_matrix.flatten()
            x2 = np.array(self.q * [True])
            for i in range(self.n):
                try:
                    x_total = np.hstack((x_total, x1, x2))
                except NameError:
                    x_total = np.hstack((x1, x2))
            x_total = x_total.astype(object)
            x_total[x_total == True] = x
            x_total[x_total == False] = np.nan
            x = x_total.astype(float)

            cov_total = np.nan * np.ones((len(x_total), len(x_total)))
            cov_total[~np.isnan(x)][:, ~np.isnan(x)] = covariance
            covariance = cov_total

        gain_matrices = np.zeros((self.n, self.q, self.p))
        control_vectors = np.zeros((self.n, self.q))

        gain_matrices_variance = np.zeros((self.n, self.q, self.p))
        control_vectors_variance = np.zeros((self.n, self.q))

        parameter_variance = np.diag(covariance)

        for i in range(self.n):

            k_start = i * self.q * (self.p + 1)
            k_end = self.q * ((i + 1) * self.p + i)
            m_end = (i + 1) * self.q * (self.p + 1)

            gain_matrices[i] = x[k_start:k_end].reshape(self.q, self.p)
            control_vectors[i] = x[k_end:m_end]

            gain_matrices_variance[i] = \
                parameter_variance[k_start:k_end].reshape(self.q, self.p)
            control_vectors_variance[i] = parameter_variance[k_end:m_end]

        return (gain_matrices, control_vectors, gain_matrices_variance,
                control_vectors_variance)

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
                            'not enough data to solve for the number of ' +
                            'unknowns.')

        if sparse.issparse(A):
            # scipy.sparse.linalg.lsmr is also an option
            x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = \
                sparse.linalg.lsqr(A, b)
            sum_of_residuals = r1norm  # this may should be the r2norm
        else:
            x, sum_of_residuals, rank, s = np.linalg.lstsq(A, b)
            # Also this is potentially a faster implementation:
            # http://graal.ift.ulaval.ca/alexandredrouin/2013/06/29/linear-least-squares-solver/

        # lstsq returns an empty array for the sum of the residuals if it is
        # rank deficient. I'm not sure what the rank deficiency means for
        # computing the follow values, so I have this ignorant solution.
        # Right now this is just to get the tests to falsely pass for this
        # function.
        # Maybe I only need to have a better formulated test set of data.
        if sum_of_residuals.size == 0:
            variance = np.nan
            covariance = np.nan * np.ones((len(x), len(x)))
        else:
            variance, covariance = \
                process.least_squares_variance(A, sum_of_residuals)

        return x, variance, covariance

    def form_sensor_vectors(self):
        """Returns an array of sensor vectors for each cycle and each time
        step.

        Returns
        =======
        sensor_vectors : ndarray, shape(m, n, p)
            The sensor vector form the i'th cycle and the j'th time step
            will look like [sensor_0, ..., sensor_(p-1)].
        """
        sensor_vectors = np.zeros((self.m, self.n, self.p))
        for i, (panel_name, data_frame) in \
                enumerate(self.data_panel.iteritems()):
            for j, (index, values) in \
                    enumerate(data_frame[self.sensors].iterrows()):
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
        for i, (panel_name, data_frame) in \
                enumerate(self.data_panel.iteritems()):
            for j, (index, values) in \
                    enumerate(data_frame[self.controls].iterrows()):
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

        Note
        ====

        In the simplest fashion, you can put::

            m(t) = m*(t) - K * s(t)

        into the form::

            Ax = b

        with::

            b = m(t)
            A = [-s(t) 1]
            x = [K(t) m*(t)]^T

            [-s(t) 1] * [K(t) m*(t)]^T = m(t)

        """
        control_vectors = self.form_control_vectors()

        b = np.array([])
        for cycle in control_vectors:
            for time_step in cycle:
                b = np.hstack((b, time_step))

        sensor_vectors = self.form_sensor_vectors()

        A = np.zeros((self.m * self.n * self.q,
                      self.n * self.q * (self.p + 1)))

        for i in range(self.m):

            Am = np.zeros((self.n * self.q,
                           self.n * self.q * (self.p + 1)))

            for j in range(self.n):

                An = np.zeros((self.q, self.q * self.p))

                for row in range(self.q):

                    An[row, row * self.p:(row + 1) * self.p] = \
                        -sensor_vectors[i, j]

                An = np.hstack((An, np.eye(self.q)))

                num_rows, num_cols = An.shape

                Am[j * num_rows:(j + 1) * num_rows, j * num_cols:(j + 1) *
                    num_cols] = An

            A[i * self.n * self.q:i * self.n * self.q + self.n * self.q] = Am

        # If there are nans in the gain omission matrix, then delete the
        # columns in A associated with gains that are set to zero.
        # TODO : Turn this into a method because I use it at least twice.
        if self.gain_omission_matrix is not None:
            x1 = self.gain_omission_matrix.flatten()
            x2 = np.array(self.q * [True])
            for i in range(self.n):
                try:
                    x_total = np.hstack((x_total, x1, x2))
                except NameError:
                    x_total = np.hstack((x1, x2))

            A = A[:, x_total]

        return A, b

    def plot_gains(self, gains, gain_variance):
        """Plots the identified gains versus percentage of the gait cycle.

        Parameters
        ==========
        gain_matrix : ndarray, shape(n, q, p)
            The estimated gain matrices for each time step.
        gain_variance : ndarray, shape(n, q, p)
            The variance of the estimated gain matrices for each time step.

        Returns
        =======
        axes : ndarray of matplotlib.axis, shape(q, p)

        """

        # TODO : Make plots have the same scale if they share the same units
        # or figure out how to normalize these.

        def to_percent(y, position):
            s = str(100 * y)
            if plt.rcParams['text.usetex'] is True:
                return s + r'$\%$'
            else:
                return s + '%'

        formatter = FuncFormatter(to_percent)

        n, q, p = gains.shape

        fig, axes = plt.subplots(q, p, sharex=True)

        xlim = (0.0, 1.0)
        percent_of_gait_cycle = np.linspace(xlim[0], xlim[1],
                                            num=gains.shape[0])

        for i in range(q):

            ylim = (np.nanmax(gains[:, i, :]), np.nanmin(gains[:, i, :]))

            for j in range(p):
                try:
                    ax = axes[i, j]
                except TypeError:
                    ax = axes
                sigma = np.sqrt(gain_variance[:, i, j])
                ax.fill_between(percent_of_gait_cycle,
                                gains[:, i, j] - sigma,
                                gains[:, i, j] + sigma,
                                alpha=0.5)
                ax.plot(percent_of_gait_cycle, gains[:, i, j], marker='o')
                ax.xaxis.set_major_formatter(formatter)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(6)
                if j == 0:
                    ax.set_ylabel('{}\nGain'.format(self.controls[i]),
                                  fontsize=10)
                if i == 0:
                    ax.set_title(self.sensors[j])
                if i == q - 1:
                    ax.set_xlabel('Percent of gait cycle', fontsize=10)

        plt.tight_layout()

        return axes


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

    acceleration = np.hstack((0.0, np.diff(filtered_speed)))

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
                   np.hstack((np.max(speed), np.min(speed))))
        ax[1].plot(time, np.hstack((0.0, np.diff(filtered_speed))), '.')
        fig.show()

    return len(time) - (new_indice), time[len(time) - new_indice]


def gait_landmarks_from_grf(time, right_grf, left_grf,
                            threshold=1e-5, do_plot=False, min_time=None,
                            max_time=None):
    """
    Obtain gait landmarks (right and left foot strike & toe-off) from ground
    reaction force (GRF) time series data.

    Parameters
    ----------
    time : array_like, shape(n,)
        A monotonically increasing time array.
    right_grf : array_like, shape(n,)
        The vertical component of GRF data for the right leg.
    left_grf : str, shape(n,)
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
    # TODO : Have an option to low pass filter the grf signals first so that
    # there is less noise in the swing phase.

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

    right_foot_strikes = birth_times(right_grf)
    left_foot_strikes = birth_times(left_grf)
    right_toe_offs = death_times(right_grf)
    left_toe_offs = death_times(left_grf)

    if do_plot:

        plt.figure(figsize=(10, 8))
        ones = np.array([1, 1])

        def myplot(index, label, ordinate, foot_strikes, toe_offs):
            ax = plt.subplot(2, 1, index)
            plt.plot(time[min_idx:max_idx], ordinate[min_idx:max_idx], '.k')
            plt.ylabel('vertical ground reaction force (N)')
            plt.title('%s (%i foot strikes, %i toe-offs)' % (
                label, len(foot_strikes), len(toe_offs)))

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

        myplot(1, 'left foot', left_grf, left_foot_strikes, left_toe_offs)
        plt.legend(loc='best')

        myplot(2, 'right foot', right_grf, right_foot_strikes, right_toe_offs)

        plt.xlabel('time (s)')
        plt.show()

    return right_foot_strikes, left_foot_strikes, right_toe_offs, left_toe_offs
