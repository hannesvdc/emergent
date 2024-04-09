"""Integrate ensemble of Stuart-Landau oscillators with linear global coupling."""

######################################################################################
#                                                                                    #
# Nov 2016                                                                           #
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


def create_initial_conditions(ic, N):
    """Specify initial conditions for zero-flux boundary conditions."""
    if ic == 'random':
        y0 = 0.5 * np.ones(N) + 0.3 * np.random.randn(N)
    if ic == 'randomabs':
        y0 = np.abs(1.0 * np.ones(N) + 0.3 * np.random.randn(N))
    elif ic == 'weakrandom':
        y0 = np.ones(N) + 0.01 * np.random.randn(N) + 0.01j * np.random.randn(N)
    elif ic == 'weakrandom_asynch':
        y0 = np.ones(N) + 0.01 * np.random.randn(N) + 0.01j * np.random.randn(N)
        y0[-int(N / 2):] = -y0[-int(N / 2):]
    elif ic == 'veryweakrandom':
        y0 = np.ones(N) + 0.001 * np.random.randn(N) + 0.001j * np.random.randn(N)
    elif ic == 'twophasesrandomized':
        y0 = np.ones(N)
        y0[0:int(N / 2)] = y0[0:int(N / 2)] - 0.3
        y0[int(N / 2):N] = y0[int(N / 2):N] + 0.3
        y0 = y0 + 0.01 * np.random.randn(N)
    elif ic == 'synchronized':
        y0 = np.ones(N)
    return y0


def f(t, y, arg_c2, arg_kre, arg_kim):
    """Temporal evolution with linear global coupling."""
    glob_lin = np.mean(y) - y
    return y - (1 + 1j * arg_c2) * abs(y)**2 * y + \
        (arg_kre + 1j * arg_kim) * glob_lin


def jac(t, y, arg_c2, arg_kre, arg_kim):
    """Calculate the jacobian evaluated at y."""
    N = y.shape[0]
    J = np.zeros((2 * N, 2 * N))
    J[:] = arg_kre / float(N)
    J[::2, 1::2] = - arg_kim / float(N)
    J[1::2, ::2] = arg_kim / float(N)
    np.fill_diagonal(J[::2, ::2],
                     (1 - arg_kre) - (np.square(np.real(y[:])) + np.square(np.imag(y[:]))
                                      ) - 2 * np.real(y[:]) * (np.real(y[:]) - arg_c2 * np.imag(y[:])
                                                               ) + arg_kre / float(N))
    np.fill_diagonal(J[1::2, 1::2], (1 - arg_kre) - (np.square(np.real(y[:])
                                                               ) + np.square(np.imag(y[:]))
                                                     ) - 2 * np.imag(y[:]) * (np.imag(y[:]) + arg_c2 * np.real(y[:])
                                                                              ) + arg_kre / float(N))
    np.fill_diagonal(J[:-1:2, 1::2], arg_kim + arg_c2 * (np.square(np.real(y[:])
                                                                   ) + np.square(np.imag(y[:]))
                                                         ) - 2 * np.imag(y[:]) * (np.real(y[:]) - arg_c2 * np.imag(y[:])
                                                                                  ) - arg_kim / float(N))
    np.fill_diagonal(J[1::2, :-1:2], -arg_kim - arg_c2 * (np.square(np.real(y[:])
                                                                    ) + np.square(np.imag(y[:]))
                                                          ) - 2 * np.real(y[:]) * (np.imag(y[:]) +
                                                                                   arg_c2 *
                                                                                   np.real(y[:])
                                                                                   ) + arg_kim / float(N))
    return J


def integrate(dynamics='', c2=0.0, kre=0.0, kim=0.0,
              N=2, tmin=500, tmax=1000, dt=0.01, T=1000, ic='weakrandom', Ainit=0, atol=10**-8, rtol=10**-8,
              nsteps=100000):
    """Integrate ensemble of Stuart-Landau oscillators with linear global coupling."""
    tstart = time()

    # Predefined dynamics are:
    # 'type 1'                  Type I Chimera
    if dynamics == 'type 1':
        print("Taking parameters for " + dynamics + " dynamics.")
        c2 = 0.58
        kre = 1.49
        kim = 1.02
    else:
        print("No predifined dynamics selected. Taking specified parameters.")

    # Write the parameters into a dictionary for future use.
    Adict = dict()
    Adict["c2"] = c2
    Adict["kre"] = kre
    Adict["kim"] = kim
    Adict["N"] = N
    Adict["tmin"] = tmin
    Adict["tmax"] = tmax
    Adict["dt"] = dt
    Adict["T"] = T
    Adict["ic"] = ic

    # Number of timesteps
    nmax = int(np.abs(float(tmax) / float(dt)))
    if T > nmax:
        raise ValueError('T is larger than the maximal number of time steps.')

    # Number of timesteps above threshold
    n_above_thresh = int(float(np.abs((tmax - tmin)) / float(dt)))
    # Threshold itself
    n_thresh = nmax - n_above_thresh
    # Every nplt'th step is plotted
    nplt = n_above_thresh / float(T)

    if ic == 'manual':
        if (Ainit.shape[0] != N):
            raise ValueError('Initial data must have the specified N dimension.')
        y0 = Ainit
    else:
        y0 = create_initial_conditions(ic, N)

    Adict["init"] = y0

    ydata = list()
    T = list()

    t0 = 0.0
    r = sp.ode(f).set_integrator('zvode', method='Adams', atol=atol, rtol=rtol, nsteps=nsteps)
    r.set_initial_value(y0, t0).set_f_params(c2, kre, kim)
    # Write initial values
    if tmin == 0:
        ydata.append(r.y)
        T.append(r.t)

    i = 0
    while r.successful() and np.abs(r.t) < np.abs(tmax):
        i = i + 1
        r.integrate(r.t + dt)
        if (i > n_thresh) and (i % nplt == 0):
            T.append(r.t)
            ydata.append(r.y)

        if i % (np.floor(nmax / 10.0)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(i)) *
                                                (float(nmax) - float(i)), 1) + ' seconds left')
            sys.stdout.flush()
    print("\n")
    Adict["data"] = np.array(ydata)
    return Adict
