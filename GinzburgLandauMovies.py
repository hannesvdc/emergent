import os
import cv2

import numpy as np
import scipy
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
    index5 = filename.find('_T=')

    c1 = float(filename[index+4:index2])
    c2 = float(filename[index2+4:index3])
    nu = float(filename[index3+4:index4])
    eta = float(filename[index4+5:index5])
    return {'c1': c1, 'c2': c2, 'nu': nu, 'eta': eta}

def _load_data(data_directory, params):
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
        if int(T) % 5 != 0:
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
    data = []
    if directory is None:
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_New_Images/'
    min_mod = 2.0
    max_mod = 0.0
    for filename in os.scandir(directory):
        if not filename.is_file() or not filename.name.endswith('.npy'):
            continue
        T = _T(filename.name)
        if T < 10.0:
            continue
        W = np.load(directory + filename.name)
        data.append((T,W))
        min_mod = min(min_mod, np.min(np.absolute(W)))
        max_mod = max(max_mod, np.max(np.absolute(W)))
    data.sort()
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
    if image_folder is None:
        image_folder = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_New_Images/'
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

    data, min_A, max_A = _load_data(data_directory, params)
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
        ax.set_title(r'$T = $'+str(t))
        plt.savefig(video_directory + '3d_proj_T=' + str(t) + '.png')
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

    fps = 5
    video = cv2.VideoWriter(directory + video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(directory, image[1])))

    cv2.destroyAllWindows()
    video.release()

def makeHistogramMovie(directory=None):
    data = []
    if directory is None:
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_New_Images/'
    min_mod = 2.0
    max_mod = 0.0
    for filename in os.scandir(directory):
        if not filename.is_file() or not filename.name.endswith('.npy'):
            continue
        T = _T(filename.name)
        W = np.load(directory + filename.name)
        data.append((T, np.absolute(W).flatten(), np.angle(W).flatten() + np.pi))
        min_mod = min(min_mod, np.min(np.absolute(W)))
        max_mod = max(max_mod, np.max(np.absolute(W)))
    print(min_mod, max_mod)
    data.sort()

    threshold = 1500
    kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    bins = [100, 100]
    edges = [[min_mod, max_mod], [0.0, 2.0*np.pi]]
    x_grid = np.linspace(min_mod, max_mod, bins[0])
    y_grid = np.linspace(0.0, 2.0*np.pi, bins[1])
    X2, Y2 = np.meshgrid(x_grid, y_grid)
    for element in data:
        print('T =', element[0], np.min(element[2]), np.max(element[2]), np.mean(element[2]))
        H, x_edges, y_edges = np.histogram2d(x=element[1], y=element[2], bins=bins, range=edges)
        print(np.min(H),np.max(H))
        print(H.shape, kernel.shape)
        H = scipy.signal.convolve2d(H, kernel, mode='same', boundary='wrap')
        H[np.where(H>threshold)] = threshold
        H = H / np.sum(H)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X2, Y2, H.T, rstride=5, cstride=5, color='orangered', edgecolors='k', lw=0.6)
        ax.set_xlabel(r'$|W|$')
        ax.set_ylabel(r'$\angle W$')
        ax.set_title(r'$T = $'+str(element[0]))
        plt.savefig(directory + 'GL_Swarm_T=' + dot_to_bar(str(round(element[0],4))) + '.png')
        plt.close()

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
    fps = 10
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
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Circular/'
        make3DPlots(directory, directory)
        make3DMovie(directory)
    elif args.run_type == '2d_video':
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Swarm/'
        make2DPlots(directory=directory)
        make2DMovie(image_folder=directory)
    elif args.run_type == 'histogram_video':
        directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Swarm/'
        makeHistogramMovie(directory=directory)
    else:
        print('Type of experiment not recognized. Returning.')
