import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib import cm

def fGinzburgLandau(W, h, c1, c2, nu):
    dWdx = (np.roll(W, 1, axis=1) - 2.0*W + np.roll(W, -1, axis=1)) / h**2
    dWdy = (np.roll(W, 1, axis=0) - 2.0*W + np.roll(W, -1, axis=0)) / h**2
    laplacian = dWdx + dWdy
    modl = np.square(np.abs(W))
    prod_term = np.multiply(modl, W)

    # Compute right-hand side
    rhs = W + (1.0 + c1*1j) * laplacian \
            - (1.0 + c2*1j) * prod_term \
            - (1.0 + nu*1j) * np.average(W) \
            + (1.0 + c2*1j) * np.average(prod_term)
    
    return rhs

"""
This function integrates the 2-dimensional complex Ginzburg-Landau
equations. We assume periodic boundaries on the domain [0,1]x[0,1].
"""
def integrateGinzburgLandauEuler(W0: np.ndarray, 
                                 h: float, 
                                 M:int, 
                                 dt: float,
                                 Tf: float, 
                                 params: dict, 
                                 tol=0.1):
    c1 = params['c1']
    c2 = params['c2']
    nu = params['nu']

    # Do time-integration
    W = np.copy(W0)
    T = 0.0
    n_print = 0.01
    time_evolution = [(T, np.copy(W))]
    while T < Tf:
        if T > n_print:
            print('\nt =', T, 'dt =', dt, np.average(W))
            n_print += 0.01
            print('Storing T =', T)
            time_evolution.append((T, np.copy(W)))

        # Euler time-stepping
        rhs = fGinzburgLandau(W, h, c1, c2, nu)
        while True:
            if lg.norm(dt*rhs / (M*M), ord=np.inf) > tol:
                dt = 0.5*dt
            else:
                W = W + dt*rhs
                T = T + dt
                dt = 1.2*dt
                break

    return W, time_evolution

def integrateGinzburgLandauRK4(W0: np.ndarray, h: float, M: int, dt: float, Tf: float, params: dict):
    c1 = params['c1']
    c2 = params['c2']
    nu = params['nu']

    # Do time-integration
    W = np.copy(W0)
    N = int(Tf / dt)
    n_print = 0.01
    for n in range(N):
        print('\nt =', n*dt, np.average(W))

        # RK4 Temporary variables
        k1 = fGinzburgLandau(W,             h, c1, c2, nu)
        k2 = fGinzburgLandau(W + 0.5*dt*k1, h, c1, c2, nu)
        k3 = fGinzburgLandau(W + 0.5*dt*k2, h, c1, c2, nu)
        k4 = fGinzburgLandau(W +     dt*k3, h, c1, c2, nu)
        rhs = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        
        # Actual time-stepping
        W = W + dt*rhs

    return W


# I assume a [0,1] x [0,1] grid with 256 grid points in each direction
# with positive (real and imaginary) random initial conditions (can be changed later)
def runGinzburgLandau():
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5}
    dt = 1.e-6   # chosen for stability (RK$ 5.e-6)
    M = 256      # from run_2d.py
    h = 1.0 / M  # from run_2d.py
    T = 10.0     # No reason at all

    rng = rd.RandomState()
    W0 = rng.uniform(low=0.0, high=1.0, size=(M,M)) + rng.uniform(low=0.0, high=1.0, size=(M,M))*1j
    W, time_evolution = integrateGinzburgLandauEuler(W0=W0, h=h, M=M, dt=dt, Tf=T, params=params)
    print(W)

    for n in range(len(time_evolution)):
        t = time_evolution[n][0]
        np.save('simulation_data/Ginzburg_Landau_Euler_T='+str(t)+'.npy', time_evolution[n][1])

def plotGinzburgLandau(W):
    angles = np.angle(W)
    cmap = cm.gist_rainbow

    color_map = cmap(angles / (2.0*np.pi))
    grid = np.linspace(0.0, 1.0, 256)
    X2, Y2 = np.meshgrid(grid, grid)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X2, Y2, np.absolute(W), facecolors=color_map, shade=False)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$|W|$')
    plt.show()

if __name__ == '__main__':
    runGinzburgLandau()