import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

def make3DPlots():
    data_directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_Simulation/'
    video_directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_3D/'

    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    M = 512

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
        print('t =', t)

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(projection='3d')
        _ = ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1, facecolors = cm.jet((phi - min_A)/(max_A - min_A)),
                       linewidth=0, antialiased=False)
        _ = ax.plot_surface(X2, Y2, phi, rstride=35, cstride=35, color='gray', linewidth=0, antialiased=False, alpha=0.4)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$W(x, y)$')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))
        ax.set_zlim(0.025, max_A + 0.1)
        ax.set_title(r'$T = $'+str(t))
        plt.savefig(video_directory + '3d_proj_T=' + str(t) + '.png')
        plt.close()

def make3DMovie():
    directory = '/Users/hannesvdc/Research_Data/emergent/Ginzburg_Landau_3D/'
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

if __name__ == '__main__':
    make3DPlots()
