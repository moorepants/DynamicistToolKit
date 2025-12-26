import numpy as np
from scipy.optimize import minimize

from dtk.process import coefficient_of_determination


class ARX():

    def __init__(self, input, output, na=1, nb=1, nk=1):
        self.u = input
        self.y = output
        self.na = na
        self.nb = nb
        self.nk = nk
        self.num_rows_to_discard = max(self.na, self.nb + self.nk)

    def form_regressor(self):
        """
        For each y grab the

        """

        self.phi = np.nan * np.ones((len(self.u), self.na + self.nb))

        self.delay = self.nk - 1

        for i, phi_of_t in enumerate(self.phi):
            if i >= self.num_rows_to_discard:
                self.phi[i] = np.hstack((-self.y[i - self.na:i],
                                          self.u[i - self.nb - self.delay:i - self.delay]))
        self.reduced_phi = self.phi[self.num_rows_to_discard:]
        self.reduced_y = self.y[self.num_rows_to_discard:]

    def form_over_determined_a_b(self):
        self.A = self.reduced_phi
        self.b = self.reduced_y

    def form_a_b(self):
        self.A = 0.0
        self.b = 0.0
        for i in range(self.reduced_phi.shape[0]):
            self.A += np.outer(self.reduced_phi[i], self.reduced_phi[i])
            self.b += self.reduced_phi[i] * self.reduced_y[i]

    def solve(self):
        if self.A.shape[0] == self.A.shape[1]:
            self.solution = np.linalg.solve(self.A, self.b)
        else:
            self.solution, residuals, rank, s = np.linalg.lstsq(self.A, self.b)

    def estimated(self):
        return np.dot(self.reduced_phi, self.solution)

    def residuals(self):
        return self.reduced_y - self.estimated()

    def coefficient_of_determination(self):
        return coefficient_of_determination(self.reduced_y, self.estimated())

    def simulate(self, u, initial_condition=0.0):
        y_sim = np.zeros_like(u)
        y_sim[:self.num_rows_to_discard] = initial_condition

        for i in range(len(y_sim)):
            if i >= self.num_rows_to_discard:
                phi = np.hstack((-y_sim[i - self.na:i],
                                 u[i - self.nb - self.delay:i - self.delay]))
                y_sim[i] = np.sum(phi * self.solution)

        return y_sim

    def find_initial_condition(self, u, guess=None):
        def cost_function(initial_condition):
            y_sim = self.simulate(u, initial_condition)
            cost = np.linalg.norm(self.y[:len(y_sim)] - y_sim)
            print('Cost is: {}'.format(cost))
            return cost
        if guess is None:
            guess = np.random.random(self.num_rows_to_discard)
        res = minimize(cost_function, guess)
        return res.x

    def print_model(self):
        denominator = 'A(q) = '
        for theta in self.solution[:self.na]:
            denominator += '{}q^-{} + '.format(theta, i)
