import numpy as np
from dtk.systemid import ARX


def test_ARX():
    na = 3
    nb = 2

    a = 0.1 * np.random.random(na)
    b = np.random.random(nb)

    theta = np.hstack((a, b))

    print(theta)

    N = 100
    time = np.linspace(0.0, 10.0, num=N)
    u = np.sin(time)
    y = np.hstack((np.zeros(na), np.array((N - na) * [np.nan], dtype=float)))

    for i in range(len(u)):
        if np.isnan(y[i]):
            previous_outputs = -a * y[i - na:i]
            previous_inputs = b * u[i - nb:i]
            y[i] = previous_outputs.sum() + previous_inputs.sum()

    arx = ARX(u, y, na=3, nb=2)
    arx.form_regressor()
    arx.form_a_b()
    arx.solve()

    print(arx.solution)

    np.testing.assert_allclose(theta, arx.solution, rtol=1e-8, atol=1e-8)

    arx.form_over_determined_a_b()
    arx.solve()

    print(arx.solution)

    np.testing.assert_allclose(theta, arx.solution, rtol=1e-8, atol=1e-8)

    arx = ARX(u[1:], y[:-1], na=3, nb=2, nk=2)
    arx.form_regressor()
    arx.form_a_b()
    arx.solve()

    print(arx.solution)

    np.testing.assert_allclose(theta, arx.solution, rtol=1e-8, atol=1e-8)
