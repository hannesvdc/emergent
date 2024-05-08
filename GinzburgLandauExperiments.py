import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import GinzburgLandau as gl

# Code to recreate Figure 1 from [https://arxiv.org/pdf/1503.04053.pdf]
def makeFigure1():
    # Run all experiments
    params_1 = {'c1': 0.2, 'c2': 0.56, 'eta': 1.5, 'eta': 0.9}
    W_1, temporal_evolution_1, temporal_slices_1 = gl.runGinzburgLandau(params=params_1, directory=None)

    params_2 = {'c1': 0.2, 'c2': 0.58, 'eta': 1.5, 'eta': 0.9}
    W_2, temporal_evolution_2, temporal_slices_2 = gl.runGinzburgLandau(params=params_2, directory=None)

    params_3 = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    W_3, temporal_evolution_3, temporal_slices_3 = gl.runGinzburgLandau(params=params_3, directory=None)

    # Do precomputations for plotting
    x, y = np.arange(W_1.shape[0]), np.arange(W_1.shape[1])
    X2, Y2 = np.meshgrid(x, y)

    _, axs = plt.subplots(3, 3)
    axs[0,0].pcolor(X2, Y2, np.absolute(W_1))
    axs[0,0].set_xlabel(r'$x$')
    axs[0,0].set_ylabel(r'$y$')
    axs[0,1].pcolor(X2, Y2, np.absolute(W_2))
    axs[0,1].set_xlabel(r'$x$')
    axs[0,1].set_ylabel(r'$y$')
    axs[0,2].pcolor(X2, Y2, np.absolute(W_3))
    axs[0,2].set_xlabel(r'$x$')
    axs[0,2].set_ylabel(r'$y$')


    re_min, im_min = -1.5, -1.5
    re_max, im_max =  1.5,  1.5
    W_lin_1 = W_1.flatten()
    W_lin_2 = W_2.flatten()
    W_lin_3 = W_3.flatten()
    for index in range(len(W_lin_1)):
        c_1 = plt.Circle((np.real(W_lin_1[index]), np.imag(W_lin_1[index])), 0.01, color='red')
        c_2 = plt.Circle((np.real(W_lin_2[index]), np.imag(W_lin_2[index])), 0.01, color='red')
        c_3 = plt.Circle((np.real(W_lin_3[index]), np.imag(W_lin_3[index])), 0.01, color='red')
        axs[2,0].add_patch(c_1)
        axs[2,1].add_patch(c_2)
        axs[2,2].add_patch(c_3)
    axs[2,0].set_xlim(re_min, re_max)
    axs[2,0].set_ylim(im_min, im_max)
    axs[2,1].set_xlim(re_min, re_max)
    axs[2,1].set_ylim(im_min, im_max)
    axs[2,2].set_xlim(re_min, re_max)
    axs[2,2].set_ylim(im_min, im_max)
    plt.show()

makeFigure1()

