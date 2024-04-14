import os
import cv2

import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib import cm

def parseT(filename, ext='.npy'):
    index = filename.find('T=')
    index2 = filename.find(ext)
    return float(filename[index+2:index2])

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
            time_evolution.append((round(T, 4), np.copy(W)))

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

"""
This function integrates the 2-dimensional complex Ginzburg-Landau
equations. We assume periodic boundaries on the domain [0,1]x[0,1].
"""
def integrateGinzburgLandauRK4(W0: np.ndarray, h: float, M: int, dt: float, Tf: float, params: dict):
    c1 = params['c1']
    c2 = params['c2']
    nu = params['nu']

    # Do time-integration
    W = np.copy(W0)
    N = int(np.ceil(Tf / dt))
    n_print = 0.001
    time_evolution = [(0.0, np.copy(W))]
    for n in range(N):
        T = n*dt
        if T >= n_print:
            print('\nt =', round(T, 4), 'dt =', dt, np.average(W))
            print('Storing T =', round(T, 4))
            n_print += 0.001
            time_evolution.append((round(T, 4), np.copy(W)))

        # RK4 Temporary variables
        k1 = fGinzburgLandau(W,             h, c1, c2, nu)
        k2 = fGinzburgLandau(W + 0.5*dt*k1, h, c1, c2, nu)
        k3 = fGinzburgLandau(W + 0.5*dt*k2, h, c1, c2, nu)
        k4 = fGinzburgLandau(W +     dt*k3, h, c1, c2, nu)
        rhs = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        
        # Actual time-stepping
        W = W + dt*rhs

    return W, time_evolution


# I assume a [0,1] x [0,1] grid with 256 grid points in each direction
# with positive (real and imaginary) random initial conditions (can be changed later)
def runGinzburgLandau():
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5}
    dt = 1.e-6   # chosen for stability (RK$ 5.e-6)
    M = 256      # from run_2d.py
    h = 1.0 / M  # from run_2d.py
    T = 10.0
    eta = 1.0    # See [https://arxiv.org/pdf/1503.04053.pdf, equation (2)]

    rng = rd.RandomState()
    W0 = rng.uniform(low=0.0, high=1.0, size=(M,M)) + rng.uniform(low=0.0, high=1.0, size=(M,M))*1j
    #W0 = rng.uniform(low=0.0, high=2.0*eta, size=(M,M)) + rng.uniform(low=-1.0, high=1.0, size=(M,M))*1j # <W0> = eta?
    W, time_evolution = integrateGinzburgLandauRK4(W0=W0, h=h, M=M, dt=dt, Tf=T, params=params)
    print(W)

    directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau/'
    for n in range(len(time_evolution)):
        t = time_evolution[n][0]
        np.save(directory + 'Ginzburg_Landau_Euler_T='+str(t)+'.npy', time_evolution[n][1])

def plotGinzburgLandau():
    # Load Data
    T = 10.0; min_dist = np.inf
    directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau/'
    for filename in os.scandir(directory):
        if not filename.is_file():
            continue
        T_file = parseT(filename.name)
        if np.abs(T_file - T) < min_dist:
            min_dist = np.abs(T_file - T)
            W = np.load(directory + filename.name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(np.abs(W))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.show()

def storeImages():
    # Load Data
    data = []
    directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_old/'
    for filename in os.scandir(directory):
        if not filename.is_file() or not filename.name.endswith('.npy'):
            continue
        T = parseT(filename.name)
        W = np.load(directory + filename.name)
        data.append((T,W))
    data.sort()

    for n in range(len(data)):
        t = data[n][0]
        W = data[n][1]
        print('t =', t)

        _ = plt.figure()
        plt.pcolor(np.abs(W) / np.max(np.abs(W)))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'$T = $'+str(t))
        plt.savefig(directory + 'GL_T='+str(round(t,4))+'.png')
        plt.close()

def makeMovie():
    image_folder = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_old/'
    video_name = 'Ginzburg_Landau.avi'

    images = []
    for img in os.listdir(image_folder):
        if not img.endswith('.png') or not img.startswith('GL'):
            continue
        T = parseT(img, ext='.png')
        images.append((T, img))
    images.sort()
        
    #images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith('GL')]
    frame = cv2.imread(os.path.join(image_folder, images[0][1]))
    height, width, layers = frame.shape

    fps = 10
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image[1])))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    runGinzburgLandau()