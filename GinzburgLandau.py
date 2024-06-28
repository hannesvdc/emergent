import argparse
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def parseT(filename, ext='.npy'):
    index = filename.find('T=')
    index2 = filename.find(ext)
    return float(filename[index+2:index2])

def dot_to_bar(string):
    return string.replace(".","_")

def bar_to_dot(string):
    return string.replace('_', '.')

def create_initial_conditions(ic, Lp, N, eta, seed=None):
    """Specify initial conditions for periodic boundary conditions. """
    binsp = (Lp / N) * np.arange(-N / 2, N / 2)
    Xp, Yp = np.meshgrid(binsp, binsp)
    if seed is not None:
        np.random.seed(seed)

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

def integrateGinzburgLandauETD2(W0, Lp, M, dt, Tf, params, T_min_store=0.0, store_slice=False):
    assert M % 2 == 0

    c1 = params['c1']
    c2 = params['c2']
    nu = params['nu']
    N = int(np.ceil(Tf / dt))

    # Frequency Space
    k = np.concatenate((np.arange(M / 2 + 1), np.arange(-M / 2 + 1, 0))) * 2.0 * np.pi / Lp # DC at 0, k > 0 first, then f < 0
    k2 = k**2
    kX, kY = np.meshgrid(k2, k2)

    # Linear Terms corresponding to W and (1 + 1c_1) nabla**2 W
    cA = 1.0 - (1.0 + c1*1j) * (kX + kY)
    cA[0,0] = -nu*1j # Subtract spatial average in DC component (<W(t=0)> = 1)
    expA = np.exp(dt * cA)

    # Nonlinear factor terms for ETD2 method
    nl_fac_A  = (expA * (1.0 + 1.0 / (cA * dt)) - 1.0 / (cA * dt) - 2.0) / cA
    nl_fac_Ap = (expA * (-1.0 / (cA * dt))      + 1.0 / (cA * dt) + 1.0) / cA

    # Do PDE Timestepping
    W = np.copy(W0)
    A = fft.fft2(W)
    temporal_evolution = list()
    temporal_slices = np.zeros((int(1 if not store_slice else (Tf - T_min_store)/dt), W.shape[1]), dtype=complex)
    slice_counter = 0
    for n in range(N):
        if n % 10 == 0:
            print('T =', n*dt)
            if n * dt >= T_min_store:
                temporal_evolution.append((n*dt, W))
            
        if store_slice and n * dt >= T_min_store:
            temporal_slices[slice_counter,:] = W[:,100]
            slice_counter += 1

        # Calculation of nonlinear part in Fourier space
        nlA = -(1 + c2 * 1j) * fft.fft2(W * np.absolute(W)**2)
        nlA[0, 0] = 0 # Subtract DC component
        if n == 0:
            nlAp = np.copy(nlA)

        # Actual Timestepping
        A = expA[:,:] * A[:,:] + nl_fac_A[:,:] * nlA[:,:] + nl_fac_Ap[:,:] * nlAp[:,:]
        W = fft.ifft2(A)

        # Update variables for next iteration. Can be with refrence because a new nlA is created every iteration
        nlAp = nlA

    temporal_evolution.append((Tf, W))
    return W, temporal_evolution, temporal_slices


""" I assume a [0,1] x [0,1] grid with 256 grid points in each direction
    with positive (real and imaginary) random initial conditions (can be changed later)
"""
def runGinzburgLandau(params={'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}, directory=None, plot=True):
    dt = 0.01    # See [https://arxiv.org/8pdf/1503.04053.pdf, Figure 1(c)]
    M = 512      # from run_2d.py
    L = 400.0    # from run_2d.py
    Lp = 2.0*L   # For some reason...
    T = 5000.0   # Need large enough timeframe for chimeras to form
    seed = 100   # Can be changed, but gives nice pictures! (100 standard)

    W0 = create_initial_conditions("plain_rand", Lp, M, params['eta'], seed=seed)
    W, temporal_evolution, temporal_slices = integrateGinzburgLandauETD2(W0=W0, 
                                                                         Lp=Lp, 
                                                                         M=M, 
                                                                         dt=dt, 
                                                                         Tf=T, 
                                                                         params=params,
                                                                         T_min_store=T-500.,
                                                                         store_slice=False)

    if directory is not None:
        print('Storing results in', directory)
        parameters_in_filename = lambda temp: '_c1='   + str(params['c1']) \
                                            + '_c2='   + str(params['c2']) \
                                            + '_nu='   + str(params['nu']) \
                                            + '_eta='  + str(params['eta']) \
                                            + '_seed=' + str(seed) \
                                            + '_T=' + str(temp)+'.npy'
        #np.save(directory + 'Ginzburg_Landau_ETD2_SS' + parameters_in_filename(T), W)
        #np.save(directory + 'Ginzburg_Landau_ETD2_Slice' + parameters_in_filename(T), temporal_slices)
        for n in range(len(temporal_evolution)):
            t = temporal_evolution[n][0]
            np.save(directory + 'Ginzburg_Landau_ETD2_Evolution' + parameters_in_filename(t), temporal_evolution[n][1])
    if plot:
        plotGinzburgLandau(W, temporal_slices, seed)
    
    return W, temporal_evolution, temporal_slices

def plotGinzburgLandau(W, temporal_slices, seed=None):
    x = np.arange(W.shape[0])
    X2, Y2 = np.meshgrid(x, x)

    # Make snapshot of |W| in xy-plane
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(X2, Y2, np.absolute(W))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Type I Chimera Seed = ' + str(seed))

    # Plot time-evolution in ty-plane
    N_timepoints = temporal_slices.shape[0]
    t = np.linspace(2000.0, 2500, N_timepoints)
    y = np.arange(W.shape[1])
    T2, Y2 = np.meshgrid(t, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(T2, Y2, np.absolute(temporal_slices).T)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Time-Evolution of Type I Chimera')

    # Show all oscillators at T = 2500 seconds
    re_min, im_min = -1.5, -1.5
    re_max, im_max =  1.5,  1.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    W_lin = W.flatten()
    centre = plt.Circle((0.0, 0.0), 1.0, color='blue', linestyle='dashed', fill = False)
    ax.add_patch(centre)
    for index in range(len(W_lin)):
        c = plt.Circle((np.real(W_lin[index]), np.imag(W_lin[index])), 0.01, color='red')
        ax.add_patch(c)
    ax.set_xlim(re_min, re_max)
    ax.set_ylim(im_min, im_max)
    plt.show()

if __name__ == '__main__':
    def parseArguments():
        parser = argparse.ArgumentParser(description='Input for the Ginzburg-Landau PDE Solver.')
        parser.add_argument('--directory', type=str, nargs='?', dest='directory', default=None, help="""
                            Name of directory to store simulation results, figures and movies. Default is not storing.
                            """)
        return parser.parse_args()

    args = parseArguments()
    runGinzburgLandau(directory=args.directory)