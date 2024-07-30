"""Integrate ensemble of Stuart-Landau oscillators."""

######################################################################################
#                                                                                    #
# doi: http://scitation.aip.org/content/aip/journal/chaos/25/6/10.1063/1.4921727     #
# Ju; 2016                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from time import time
import sys
import scipy.integrate as sp

#############################################################################
# INTEGRATION
#############################################################################


def create_initial_conditions(ic, N, eta):
    """Specify initial conditions for zero-flux boundary conditions."""
    if ic == 'random':
        y0 = 0.5 * np.ones(N) + 0.3 * np.random.randn(N)
        y0 = (y0 / np.mean(y0)) * eta
    if ic == 'randomabs':
        y0 = np.abs(1.0 * np.ones(N) + 0.3 * np.random.randn(N))
        y0 = (y0 / np.mean(y0)) * eta
    elif ic == 'weakrandom':
        y0 = eta * np.ones(N) + 0.01 * np.random.randn(N)
        y0 = (y0 / np.mean(y0)) * eta
    elif ic == 'veryweakrandom':
        y0 = eta * np.ones(N) + 0.001 * np.random.randn(N)
        y0 = (y0 / np.mean(y0)) * eta
    elif ic == 'twophasesrandomized':
        y0 = eta * np.ones(N)
        y0[0:N / 2] = y0[0:N / 2] - 0.3
        y0[N / 2:N] = y0[N / 2:N] + 0.3
        y0 = y0 + 0.01 * np.random.randn(N)
        y0 = (y0 / np.mean(y0)) * eta
    elif ic == 'synchronized':
        y0 = eta * np.ones(N)
        y0 = (y0 / np.mean(y0)) * eta
    return y0


def f(t, y, arg_c2, arg_nu, arg_N):
    """Temporal evolution with non-linear global coupling."""
    glob_lin = np.ones(arg_N) * np.mean(y)
    glob_nonlin = np.ones(arg_N) * np.mean(abs(y)**2 * y)
    return y - (1 + 1j * arg_c2) * abs(y)**2 * y - \
        (1 + 1j * arg_nu) * glob_lin + (1 + 1j * arg_c2) * glob_nonlin


def integrate(dynamics='type 1', c2=0.58, nu=1.49, eta=1.02,
              N=512, tmin=500, tmax=1000, dt=0.01, T=1000, ic='random', Ainit=0):
    """Integrate ensemble of Stuart-Landau oscillators with non-linear global coupling."""
    tstart = time()

    # Predefined dynamics are:
    # 'type 1'                  Type I Chimera
    # 'type 2'                  Type II Chimera
    # 'mod-amp-cluster'         Modulated-amplitude Cluster
    if dynamics == 'type 1':
        print("Taking parameters for " + dynamics + " dynamics.")
        c2 = 0.58
        nu = 1.49
        eta = 1.021
    elif dynamics == 'type 2':
        print("Taking parameters for " + dynamics + " dynamics.")
        c2 = -0.66
        nu = 0.1
        eta = 0.67
    elif dynamics == 'mod-amp-cluster':
        print("Taking parameters for " + dynamics + " dynamics.")
        c2 = -0.6
        nu = 0.1
        eta = 0.7
    else:
        print("No predifined dynamics selected. Taking specified parameters.")

    # Write the parameters into a dictionary for future use.
    Adict = dict()
    Adict["c2"] = c2
    Adict["nu"] = nu
    Adict["eta"] = eta
    Adict["N"] = N
    Adict["tmin"] = tmin
    Adict["tmax"] = tmax
    Adict["dt"] = dt
    Adict["T"] = T
    Adict["ic"] = ic

    # Number of timesteps
    nmax = int(tmax / dt)
    if T > nmax:
        raise ValueError('T is larger than the maximal number of time steps.')

    # Number of timesteps above threshold
    n_above_thresh = int((tmax - tmin) / dt)
    # Threshold itself
    n_thresh = nmax - n_above_thresh
    # Every nplt'th step is plotted
    nplt = n_above_thresh / T

    if ic == 'manual':
        if (Ainit.shape[0] != N):
            raise ValueError('Initial data must have the specified N dimension.')
        y0 = Ainit
    else:
        y0 = create_initial_conditions(ic, N, eta)

    Adict["init"] = y0

    ydata = list()
    T = list()

    t0 = 0
    r = sp.ode(f).set_integrator('zvode', method='Adams')
    r.set_initial_value(y0, t0).set_f_params(c2, nu, N)
    # Write initial values
    if tmin == 0:
        ydata.append(r.y)
        T.append(r.t)

    i = 0
    while r.successful() and r.t < tmax:
        i = i + 1
        r.integrate(r.t + dt)
        if (i > n_thresh) and (i % nplt == 0):
            T.append(r.t)
            ydata.append(r.y)

        if i % (np.floor(nmax / 10.0)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(i)) *
                                                (float(nmax) - float(i)), 1) + ' seconds left')
            sys.stdout.flush()
    Adict["data"] = np.array(ydata)
    return Adict
