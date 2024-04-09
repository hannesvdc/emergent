"""Integration of the modified Ginzburg-Landau equation with 2 spatial dimensions."""

######################################################################################
#                                                                                    #
# doi: http://scitation.aip.org/content/aip/journal/chaos/25/6/10.1063/1.4921727     #
#                                                                                    #
# Mar 2016                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
import pylab as pl
from time import time
import sys

#############################################################################
# INTEGRATION
#############################################################################


def create_initial_conditions(ic, Lp, N, eta):
    """Specify initial conditions for zero-flux boundary conditions."""
    print("Creating initial conditions (" + ic + ").")
    binsp = (Lp / N) * np.arange(-N / 2, N / 2)
    Xp, Yp = np.meshgrid(binsp, binsp)
    if ic == 'pulse':
        A = 0.5 + 0.5 * (1 / np.cosh((Xp**2 + Yp**2) * 0.2)) + \
            10**(-2) * np.random.randn(len(Xp), len(Yp))
        A = (A * N**2 / np.sum(A)) * eta
    if ic == 'sine1D':
        A = 0.01 * np.sin(2 * np.pi * 40 * Yp / Lp) + 0.5
        A = (A * N**2 / np.sum(A)) * eta
    if ic == 'sine2D':
        A = 0.01 * np.sin(2 * np.pi * 40 * Xp / Lp) + 0.01 * np.sin(2 * np.pi * 40 * Yp / Lp) + 0.5
        A = (A * N**2 / np.sum(A)) * eta
    if ic == 'plain':
        A = Xp * 0.0 + 0.5
        A = (A * N**2 / np.sum(A)) * eta
    if ic == 'plain_rand':
        A = Xp * 0.0 + 0.5
        A[:int(N / 4), :] = A[:int(N / 4), :] + 0.01 * \
            np.reshape(np.random.randn(int(N / 4 * N)), (int(N / 4), N))
        A = (A * N**2 / np.sum(A)) * eta
    return A


def pad_reflect_2D(array, N):
    """Create NxN data matrix from N/2xN/2 matrix with zero flux boundaries."""
    # deltaY/deltaX should not exceed length of corresponding dimension of array.
    deltax = int(N / 4)
    # output = np.zeros((int(len(array) + 2 * deltax),
    #                    int(len(array[0]) + 2 * deltax))).astype(np.complex256)
    output = np.zeros((N, N), dtype='complex')
    output[deltax:-deltax, deltax:-deltax] = array
    output[0:deltax] = output[2 * deltax:deltax:-1]
    output[-deltax:] = output[-deltax - 2:-2 * deltax - 2:-1]
    output[:, 0:deltax] = output[:, 2 * deltax:deltax:-1]
    output[:, -deltax:] = output[:, -deltax - 2:-2 * deltax - 2:-1]
    return output.astype(np.complex128)


def integrate(dynamics='type 1', c1=0.0, c2=0.0, nu=0.0, eta=0.0,
              L=100., N=256, tmin=500,
              tmax=1000, dt=0.05, T=1000, bc='no-flux', ic='plain_rand', Ainit=0):
    """Integrate MCGLE and return dictionary Adict."""
    # Predefined dynamics are:
    # 'type 1'                  Type I Chimera
    # 'type 2'                  Type II Chimera
    # 'localized turbulence'    Localized Turbulence
    # 'alternating'             Alternating Chimera
    # 'mod-amp-cluster'         Modulated-amplitude Cluster
    # 'amp-cluster'             Amplitude Cluster
    # 'synch-wave-coex'         Synchronization and Waves Coexistence patterns

    print("Integrating MCGLE:")
    if dynamics == 'type 1':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.2
        c2 = 0.61
        nu = 1.5
        eta = 1.0
    elif dynamics == 'type 2':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.2
        c2 = -0.63
        nu = 0.1
        eta = 0.65
    elif dynamics == 'localized turbulence':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = -1.6
        c2 = 1.5
        nu = 1.5
        eta = 0.9
    elif dynamics == 'alternating':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.2
        c2 = -0.67
        nu = 0.1
        eta = 0.65
    elif dynamics == 'mod-amp-cluster':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.2
        c2 = -0.58
        nu = 1.0
        eta = 0.67
    elif dynamics == 'amp-cluster':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.2
        c2 = 0.56
        nu = 1.5
        eta = 0.9
    elif dynamics == 'synch-wave-coex':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.2
        c2 = 0.585
        nu = 1.5
        eta = 0.9
    else:
        print("No predifined dynamics selected. Taking specified parameters.")

    # Write the parameters into a dictionary for future use.
    Adict = dict()
    Adict["c1"] = c1
    Adict["c2"] = c2
    Adict["nu"] = nu
    Adict["eta"] = eta
    Adict["L"] = L
    Adict["N"] = N
    Adict["tmin"] = tmin
    Adict["tmax"] = tmax
    Adict["dt"] = dt
    Adict["T"] = T
    Adict["ic"] = ic
    tstart = time()
    Lp = 2 * L
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

    binsp = (Lp / N) * np.arange(-N / 2, N / 2)
    Xp, Yp = np.meshgrid(binsp, binsp)

    # Set of wavenumbers
    k = np.concatenate((np.arange(N / 2 + 1), np.arange(-N / 2 + 1, 0))) * 2 * np.pi / Lp
    k2 = k**2
    k2X, k2Y = np.meshgrid(k2, k2)

    # data lists
    # data in physical space
    Adata = list()
    # Time Array
    T = list()

    # Set initial conditions
    if ic == 'manual':
        print("Using the given data as initial conditions.")
        A = Ainit
        if bc == 'periodic':
            if ((Ainit.shape[0] != N) or (Ainit.shape[1] != N)):
                raise ValueError('Initial data must have the specified NxN dimension.')
            A_hat = np.fft.fft2(A)
        elif bc == 'no-flux':
            if ((Ainit.shape[0] != N / 2) or (Ainit.shape[1] != N / 2)):
                raise ValueError('Initial data must have the specified N/2xN/2 dimension.')
            # Make a NxN matrix out of the N/2xN/2 data through mirroring.
            A = pad_reflect_2D(A, N)
            A_hat = np.fft.fft2(A)
            # No-flux boundary conditions
            A_hat[0:int(N / 2 + 1), int(N / 2 + 1):N][:, ::2] = - \
                A_hat[0:int(N / 2 + 1), 1:int(N / 2)][:, ::-1][:, ::2]
            A_hat[0:int(N / 2 + 1), int(N / 2 + 2):N][:, ::2] = A_hat[0:int(N / 2 + 1),
                                                                      1:int(N / 2 - 1)][:, ::-1][:, ::2]
            # rest...
            A_hat[int(N / 2 + 1):N, :][::2, :] = -A_hat[1:int(N / 2), :][::-1, :][::2, :]
            A_hat[int(N / 2 + 2):N, :][::2, :] = A_hat[1:int(N / 2 - 1), :][::-1, :][::2, :]
            A = np.fft.ifft2(A_hat)
        else:
            raise ValueError("Please set proper boundary conditions: 'periodic' or 'no-flux'")
    else:
        A = create_initial_conditions(ic, Lp, N, eta)
        A_hat = np.fft.fft2(A)
        if bc == 'no-flux':
            # No-flux boundary conditions
            A_hat[0:int(N / 2 + 1), int(N / 2 + 1):N][:, ::2] = - \
                A_hat[0:int(N / 2 + 1), 1:int(N / 2)][:, ::-1][:, ::2]
            A_hat[0:int(N / 2 + 1), int(N / 2 + 2):N][:, ::2] = A_hat[0:int(N / 2 + 1),
                                                                      1:int(N / 2 - 1)][:, ::-1][:, ::2]
            # rest...
            A_hat[int(N / 2 + 1):N, :][::2, :] = -A_hat[1:int(N / 2), :][::-1, :][::2, :]
            A_hat[int(N / 2 + 2):N, :][::2, :] = A_hat[1:int(N / 2 - 1), :][::-1, :][::2, :]
        elif bc == 'periodic':
            pass
        else:
            raise ValueError("Please set proper boundary conditions: 'periodic' or 'no-flux'")
        A = np.fft.ifft2(A_hat)

    Adict["init"] = A

    if n_thresh == 0:
        # Take only half of the data points
        Adata.append(A[int(N / 4):int(3 * N / 4), int(N / 4):int(3 * N / 4)])
        T.append(0)

    # Compute exponentials and nonlinear factors for ETD2 method
    cA = 1 - (k2X + k2Y) * (1 + c1 * 1j)
    # Homogeneous mode
    cA[0, 0] = -nu * 1j
    expA = np.exp(dt * cA)
    nlfacA = (np.exp(dt * cA) * (1 + 1 / (cA * dt)) - 1 / (cA * dt) - 2) / cA
    nlfacAp = (np.exp(dt * cA) * (-1 / (cA * dt)) + 1 / (cA * dt) + 1) / cA

    # Solve PDE
    for i in np.arange(1, nmax + 1):
        # Calculation of nonlinear part in Fourier space
        nlA = -(1 + c2 * 1j) * pl.fft2(A * abs(A)**2)
        # Homogeneous mode
        nlA[0, 0] = 0

        # Setting the first values of the previous nonlinear coefficients
        if i == 1:
            nlAp = nlA

        # Time-stepping
        if bc == 'no-flux':
            A_hat[0:int(N / 2 + 1), 0:int(N / 2 + 1)] = \
                A_hat[0:int(N / 2 + 1), 0:int(N / 2 + 1)] * expA[0:int(N / 2 + 1), 0:int(N / 2 + 1)] + \
                nlfacA[0:int(N / 2 + 1), 0:int(N / 2 + 1)] * nlA[0:int(N / 2 + 1), 0:int(N / 2 + 1)] + \
                nlfacAp[0:int(N / 2 + 1), 0:int(N / 2 + 1)] * \
                nlAp[0:int(N / 2 + 1), 0:int(N / 2 + 1)]
            # No-flux boundary conditions
            A_hat[0:int(N / 2 + 1), int(N / 2 + 1):N][:, ::2] = - \
                A_hat[0:int(N / 2 + 1), 1:int(N / 2)][:, ::-1][:, ::2]
            A_hat[0:int(N / 2 + 1), int(N / 2 + 2):N][:, ::2] = \
                A_hat[0:int(N / 2 + 1), 1:int(N / 2 - 1)][:, ::-1][:, ::2]
            # rest...
            A_hat[int(N / 2 + 1):N, :][::2, :] = -A_hat[1:int(N / 2), :][::-1, :][::2, :]
            A_hat[int(N / 2 + 2):N, :][::2, :] = A_hat[1:int(N / 2 - 1), :][::-1, :][::2, :]
        elif bc == 'periodic':
            # Time-stepping (carried out in parallel for each individual Fourier mode)
            A_hat[:] = A_hat[:] * expA[:] + nlfacA[:] * nlA[:] + nlfacAp[:] * nlAp[:]

        A = pl.ifft2(A_hat)
        nlAp = nlA

        # Saving data
        if (i > n_thresh) and (i % nplt == 0):
            # again only half of the data
            if bc == 'no-flux':
                # again only half of the data
                Adata.append(A[int(N / 4):int(3 * N / 4), int(N / 4):int(3 * N / 4)])
            elif bc == 'periodic':
                Adata.append(A)
            T.append(i * dt)

        if i % (np.floor(nmax / 100)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(i)) *
                                                (float(nmax) - float(i)), 1) + ' seconds left')
            sys.stdout.flush()

    print("\n")
    tend = time()
    print('Simulation completed!')
    print('Running time: ', round((tend - tstart), 1), 's')
    Adict["data"] = np.array(Adata)
    return Adict
