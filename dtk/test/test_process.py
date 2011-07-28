from dtk.process import *
import numpy as np
import matplotlib.pyplot as plt

def test_spline_over_nan():
    x = np.linspace(0., 50., num=300)
    y = np.sin(x) + np.random.rand(len(x))
    # add some nan's
    y[78:89] = np.nan
    y[395:455] = np.nan
    y[0] = np.nan
    y[212] = np.nan

    ySplined = spline_over_nan(x, y)
    #plt.plot(x, ySplined)
    #plt.plot(x, y)
    #plt.show()
