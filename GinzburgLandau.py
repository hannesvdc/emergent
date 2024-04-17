import os
import cv2
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm

def parseT(filename, ext='.npy'):
    index = filename.find('T=')
    index2 = filename.find(ext)
    return float(filename[index+2:index2])

def create_initial_conditions(ic, Lp, N, eta, seed=None):
    """Specify initial conditions for zero-flux boundary conditions."""
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

def integrateGinzburgLandauETD2(W0, Lp, M, dt, Tf, params):
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
    for n in range(N):
        if n % 100 == 0:
            print('T =', n*dt, np.min(np.absolute(W)), np.max(np.absolute(W)))
            temporal_evolution.append((n*dt, np.copy(W)))

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

    temporal_evolution.append((Tf, np.copy(W)))
    return W, temporal_evolution


""" I assume a [0,1] x [0,1] grid with 256 grid points in each direction
    with positive (real and imaginary) random initial conditions (can be changed later)
"""
def runGinzburgLandau():
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5}
    dt = 0.05    # See [https://arxiv.org/pdf/1503.04053.pdf, Figure 1(c)]
    M = 512      # from run_2d.py
    L = 400.0    # from run_2d.py
    T = 2500.0   # Need large enough timeframe for chimera's to form
    eta = 1.0    # See [https://arxiv.org/pdf/1503.04053.pdf, equation (2)]
    storeSolution = False
    seed = 100

    Lp = 2.0*L
    W0 = create_initial_conditions("plain_rand", Lp, M, eta, seed=seed)
    W, temporal_evolution = integrateGinzburgLandauETD2(W0=W0, 
                                                        Lp=Lp, 
                                                        M=M, 
                                                        dt=dt, 
                                                        Tf=T, 
                                                        params=params)
    print(W)

    plotGinzburgLandau(W)
    if storeSolution:
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau/'
        for n in range(len(temporal_evolution)):
            t = temporal_evolution[n][0]
            np.save(directory + 'Ginzburg_Landau_ETD2_T='+str(t)+'.npy', temporal_evolution[n][1])

def plotGinzburgLandau(W):
    # Load Data
    x = np.arange(W.shape[0])
    X2, Y2 = np.meshgrid(x, x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(X2, Y2, np.absolute(W))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Type I Chimera')
    plt.show()

def storeImages():
    def dot_to_bar(string):
        return string.replace(".","_")
    # Load Data
    data = []
    data_directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau/'
    store_directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Images/'
    min_mod = 2.0
    max_mod = 0.0
    for filename in os.scandir(data_directory):
        if not filename.is_file() or not filename.name.endswith('.npy'):
            continue
        T = parseT(filename.name)
        W = np.load(data_directory + filename.name)
        data.append((T,W))
        min_mod = min(min_mod, np.min(np.absolute(W)))
        max_mod = max(max_mod, np.max(np.absolute(W)))
    data.sort()
    print(min_mod, max_mod)

    M = 512
    x_grid = np.linspace(0.0, 1.0, M)
    y_grid = np.linspace(0.0, 1.0, M)
    X2, Y2 = np.meshgrid(x_grid, y_grid)
    for n in range(len(data)):
        t = data[n][0]
        W = data[n][1]
        phi = np.absolute(W)
        print('t =', t)

        _ = plt.figure()
        plt.pcolor(X2, Y2, phi, vmin=min_mod, vmax=max_mod)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'$T = $'+str(t))
        plt.savefig(store_directory + 'GL_T='+dot_to_bar(str(round(t,4)))+'.png')
        plt.close()

def makeMovie():
    def bar_to_dot(string):
        return string.replace("_", ".")
    image_folder = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Images/'
    video_name = 'Ginzburg_Landau.avi'

    images = []
    for img in os.listdir(image_folder):
        if not img.endswith('.png') or not img.startswith('GL'):
            continue
        T = parseT(bar_to_dot(img), ext='.png')
        images.append((T, img))
    images.sort()
        
    frame = cv2.imread(os.path.join(image_folder, images[0][1]))
    height, width, layers = frame.shape

    fps = 10
    video = cv2.VideoWriter(image_folder + video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image[1])))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    def parseArguments():
        parser = argparse.ArgumentParser(description='Input for the Ginzburg-Landau PDE Solver.')
        parser.add_argument('--type', type=str, nargs='?', dest='run_type', help="""Which type of experiment to run.\n
                            \tpde: Run the 2-dimensional Ginzburg-Landau PDE,\n
                            \tvideo: Make a (x,y,t) video based on the simulation data,\n
                            \thist: Make emergent-space video.
                            """)
        return parser.parse_args()

    args = parseArguments()
    if args.run_type == 'pde':
        runGinzburgLandau()
    elif args.run_type == 'video':
        #storeImages()
        makeMovie()
    elif args.run_type == 'hist':
        print('This type of experiment is currently not supported.')
    else:
        print('Type of experiment not recognized. Returning.')