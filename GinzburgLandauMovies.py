import os
import cv2

import numpy as np
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

def dot_to_bar(string):
    return string.replace(".","_")

def bar_to_dot(string):
    return string.replace('_', '.')

def _T(filename, ext='.npy'):
    index = filename.find('T=')
    index2 = filename.find(ext)
    return float(filename[index+2:index2])

def _parameters(filename):
    index = filename.find('_c1=')
    index2 = filename.find('_c2=')
    index3 = filename.find('_nu=')
    index4 = filename.find('_eta=')
    if filename.find('_seed=') != -1:
        index5 = filename.find('_seed=')
    else:
        index5 = filename.find('_T=')

    c1 = float(filename[index+4:index2])
    c2 = float(filename[index2+4:index3])
    nu = float(filename[index3+4:index4])
    eta = float(filename[index4+5:index5])
    return {'c1': c1, 'c2': c2, 'nu': nu, 'eta': eta}

def _load_data(data_directory, params, reduce=False):
    min_A = np.inf
    max_A = 0
    data = []
    for filename in os.scandir(data_directory):
        if not filename.is_file() \
           or not filename.name.startswith('Ginzburg_Landau_ETD2_Evolution') \
           or not filename.name.endswith('.npy'):
            continue
        if _parameters(filename.name) != params:
            continue
        T = _T(filename.name)
        if reduce and not T.is_integer():
            continue

        print(filename.name)

        W = np.load(data_directory + filename.name)
        data.append((T,W))

        min_A = min(min_A, np.min(np.absolute(W)))
        max_A = max(max_A, np.max(np.absolute(W)))
    data.sort()
    
    return data, min_A, max_A

def make2DPlots(directory=None):
    # Load Data
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    M = 512

    data, min_mod, max_mod = _load_data(directory, params, reduce=False)
    print('min/max mod', min_mod, max_mod)

    M = 512
    grid = np.linspace(0.0, 1.0, M)
    X2, Y2 = np.meshgrid(grid, grid)
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
        plt.savefig(directory + 'GL_T='+dot_to_bar(str(round(t,4)))+'.png')
        plt.close()

def make2DMovie(image_folder=None):
    video_name = 'Ginzburg_Landau.avi'

    images = []
    for img in os.listdir(image_folder):
        if not img.endswith('.png') or not img.startswith('GL'):
            continue
        T = _T(bar_to_dot(img), ext='.png')
        images.append((T, img))
    images.sort()
        
    frame = cv2.imread(os.path.join(image_folder, images[0][1]))
    height, width, _ = frame.shape

    fps = 10
    video = cv2.VideoWriter(image_folder + video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image[1])))

    cv2.destroyAllWindows()
    video.release()

def make3DPlots(data_directory, video_directory):
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    M = 512

    data, min_A, max_A = _load_data(data_directory, params, reduce=True)
    print('min/max mod', min_A, max_A)

    # Do actual plotting
    x_grid = np.linspace(0.0, 1.0, M)
    y_grid = np.linspace(0.0, 1.0, M)
    X2, Y2 = np.meshgrid(x_grid, y_grid)
    Z2 = np.zeros_like(X2)
    for n in range(len(data)):
        t = data[n][0]
        W = data[n][1]
        phi = np.absolute(W)
        sigma_x = sigma_y = 5
        sigma = [sigma_y, sigma_x]
        y = filters.gaussian_filter(phi, sigma, mode='wrap')
        print('t =', t)

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(projection='3d')
        _ = ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1, facecolors = cm.jet((phi - min_A)/(max_A - min_A)),
                       linewidth=0, antialiased=False)
        _ = ax.plot_surface(X2, Y2, y, rstride=5, cstride=5, color='dodgerblue', linewidth=0, alpha=0.7)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$W(x, y)$')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))
        ax.grid(False)
        ax.set_zlim(0.025, max_A + 0.1)
        ax.set_title(r'$T = $'+str(round(t,1)))
        plt.savefig(video_directory + '3d_proj_T=' + str(round(t, 1)) + '.png')
        plt.close()

def make3DMovie(directory):
    video_name = 'Ginzburg_Landau_3D.avi'

    images = []
    for img in os.listdir(directory):
        if not img.endswith('.png') or not img.startswith('3d_proj'):
            continue
        T = _T(img, ext='.png')
        images.append((T, img))
    images.sort()
        
    frame = cv2.imread(os.path.join(directory, images[0][1]))
    height, width, _ = frame.shape

    fps = 10
    video = cv2.VideoWriter(directory + video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(directory, image[1])))

    cv2.destroyAllWindows()
    video.release()

def makeHistogramMovie(directory):
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    M = 512
    dx = 1.0 / M

    # Load data and Compute D = W_xx + W_yy
    data, min_mod, max_mod = _load_data(directory, params)
    D_data = list()
    min_angle, max_angle = -np.pi, np.pi
    for (T, W) in data:
        Wxx = (np.roll(W, -1, axis=0) - 2.0*W + np.roll(W, 1, axis=0))
        Wyy = (np.roll(W, -1, axis=1) - 2.0*W + np.roll(W, 1, axis=1))
        D = (Wxx + Wyy) / dx**2

        D_angle = np.angle(D)
        min_angle = min(min_angle, np.min(D_angle))
        max_angle = max(max_angle, np.max(D_angle))
        D_data.append((T, D_angle))

    bins = [100, 100]
    edges = [[min_mod, max_mod], [min_angle, max_angle]]
    x_grid = np.linspace(min_mod, max_mod, bins[0])
    y_grid = np.linspace(min_angle, max_angle, bins[1])
    X2, Y2 = np.meshgrid(x_grid, y_grid)
    real_x_grid = np.linspace(-0.5, 0.5, bins[0])
    real_y_grid = np.linspace(-0.5, 0.5, bins[1])
    for index in range(len(data)):
        print('T = ', data[index][0])
        T = data[index][0]
        W = data[index][1]
        D = D_data[index][1]

        # Buidl histogram and artificially increase frequency of non-DC components
        H, _, _ = np.histogram2d(np.absolute(W).flatten(), D.flatten(), bins=bins, range=edges)
        H = filters.gaussian_filter(H, axes=0, sigma=0.2, mode='wrap')
        H[0:90, :] = 100.0 * H[0:90, :] # Scale for visualization, does not scale underlying data

        # Create facecolors for complex plotting
        centred_x, centred_y = np.meshgrid(real_x_grid, real_y_grid)
        phi = np.arctan2(centred_y, centred_x)

        # Do the actual plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Y2, X2, np.zeros_like(X2) - 10000, facecolors = cm.hsv((phi + np.pi)/(2.0 * np.pi)))
        ax.plot_surface(Y2, X2, H.T, color='gray', alpha=0.6)
        ax.set_xlabel(r'$\angle \Delta W$')
        ax.set_ylabel(r'$|W|$')
        ax.set_zlabel(r'$g$')
        ax.set_title(r'$T = $'+str(T))
        ax.set_zlim(-10000, 25000)
        ax.set_zticklabels([])
        ax.grid(False)
        plt.savefig(directory + 'GL_Histogram_T=' + dot_to_bar(str(round(T,4))) + '.png')
        plt.close()

    images = []
    for img in os.listdir(directory):
        if not img.endswith('.png') or not img.startswith('GL_Histogram'):
            continue
        T = _T(bar_to_dot(img), ext='.png')
        images.append((T, img))
    images.sort()
        
    frame = cv2.imread(os.path.join(directory, images[0][1]))
    height, width, _ = frame.shape
    video_name = 'Ginzburg_Landau_Histogram.avi'
    fps = 20
    video = cv2.VideoWriter(directory + video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(directory, image[1])))

    cv2.destroyAllWindows()
    video.release()

def makeSwarmPlots(data_directory):
    # Load simulation data into memory
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    data, min_A, max_A = _load_data(data_directory, params, reduce=True)

    # Compute finite differences approximation of the gradient for all T
    M = 512
    dx = 1.0 / M
    min_Re = np.inf
    max_Re = -np.inf
    min_Im = np.inf
    max_Im = -np.inf

    num_samples = 50000
    idx = np.random.choice(np.arange(M**2), num_samples)
    Re_D_data = np.zeros((num_samples, len(data)))
    Im_D_data = np.zeros((num_samples, len(data)))
    A_W_data = np.zeros((num_samples, len(data)))

    counter = 0
    for (T, W) in data:
        print('T =', T)
        Wxx = (np.roll(W, -1, axis=0) - 2.0*W + np.roll(W, 1, axis=0))
        Wyy = (np.roll(W, -1, axis=1) - 2.0*W + np.roll(W, 1, axis=1))
        D = (Wxx + Wyy) / dx**2

        RD = np.real(D)
        ID = np.imag(D)
        A = np.absolute(W)
        Re_D_data[:,counter] = RD.flatten()[idx]
        Im_D_data[:,counter] = ID.flatten()[idx]
        A_W_data[:,counter]  = A.flatten()[idx]
        counter += 1

        min_Re = min(min_Re, np.min(RD))
        max_Re = max(max_Re, np.max(RD))
        min_Im = min(min_Im, np.min(ID))
        max_Im = max(max_Im, np.max(ID))
    print(min_Re, max_Re, min_Im, max_Im)

    # Plotting
    for n in range(len(data)):
        T = data[n][0]
        print('T =', T)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.scatter(Re_D_data[:,n], Im_D_data[:,n], A_W_data[:,n], s=0.1, cmap='viridis')
        ax.scatter(Im_D_data[:,n], A_W_data[:,n], zdir='x', zs=-10.**5, color='gray', s=0.1)
        ax.scatter(Re_D_data[:,n], A_W_data[:,n], zdir='y', zs=10.**5, color='gray', s=0.1)
        ax.scatter(Re_D_data[:,n], Im_D_data[:,n], zdir='z', zs=min_A, color='gray',s=0.1)
        ax.set_xlim((-10.**5, 10.**5))
        ax.set_ylim((-10.**5, 10.**5))
        ax.set_zlim3d((min_A, max_A))
        ax.set_xlabel(r'Re $\Delta W$')
        ax.set_ylabel(r'Im $\Delta W$')
        ax.set_zlabel(r'$|W|$')
        ax.set_title(r'$T = $' + str(round(T, 1)))
        plt.savefig(directory + 'GL_Swarm_T=' + dot_to_bar(str(round(T,4))) + '.png')
        plt.close()

def makeSwarmMovie(directory):
    images = []
    for img in os.listdir(directory):
        if not img.endswith('.png') or not img.startswith('GL_Swarm'):
            continue
        T = _T(bar_to_dot(img), ext='.png')
        images.append((T, img))
    images.sort()
        
    frame = cv2.imread(os.path.join(directory, images[0][1]))
    height, width, _ = frame.shape
    video_name = 'Ginzburg_Landau_Swarm.avi'
    fps = 20
    video = cv2.VideoWriter(directory + video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(directory, image[1])))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    def parseArguments():
        parser = argparse.ArgumentParser(description='Input for the Ginzburg-Landau PDE Solver.')
        parser.add_argument('--type', type=str, nargs='?', dest='run_type', help="""Which type of visualisation to make.\n
                            \t3d_video: Make a (x,y,|W|) video based on the simulation data,\n
                            \t2d_video: Make a (x,y,t) video based on the simulation data,\n
                            \thistogram_video: Make emergent-space video,\n
                            \tswarm_video: Maka an evolving swarm video. Currently not yet implemented.\n
                            """)
        parser.add_argument('--directory', type=str, nargs='?', dest='directory', default=None, help="""
                            Name of directory to store simulation results, figures and movies. Default is not storing.
                            """)
        return parser.parse_args()
    args = parseArguments()
    
    if args.run_type == '3d_video':
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_3D/'
        make3DPlots(directory, directory)
        make3DMovie(directory)
    elif args.run_type == '2d_video':
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Simulation/'
        make2DPlots(directory=directory)
        make2DMovie(image_folder=directory)
    elif args.run_type == 'histogram_video':
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_3D/'
        makeHistogramMovie(directory=directory)
    elif args.run_type == 'swarm_movie':
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_3D/'
        makeSwarmPlots(directory)
        makeSwarmMovie(directory)
    else:
        print('Type of experiment not recognized. Returning.')
