import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import int.mcgle_2d as mint
import fun.stencil_2d as stl


def main_amp():
    """Apply discrete Laplacian on the data and create histogram of its amplitude."""
    try:
        A = np.load('data/mcgle_data.npy')
    except:
        print('Simulating MCGLE.')
        Ad = mint.integrate(L=400, N=512, tmin=2000, tmax=2500, bc='periodic')
        A = Ad["data"]
        np.save('data/mcgle_data.npy', A)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(np.abs(A[-1]))
    ax.set_xlabel('x')
    ax.set_title('Run 2D')

    ax.set_ylabel('y')
    plt.savefig('')
    plt.show()

    (T, N, N) = A.shape
    print(T, N)
    tt = np.linspace(20000, 21000, T)
    xx = np.linspace(0, 400, N)
    X, Y = np.meshgrid(xx, xx)

    stcl = stl.create_stencil(N, N, 1)

    # set parameters
    nbins = 40
    Dmax = 0.15

    AS = np.zeros((T, N, N), dtype='complex')
    for x in range(0, T):
        AS[x, :, :] = np.reshape(np.abs(stcl.dot(np.reshape(A[x, :, :], N*N))), (N, N)) + \
            1.0j*np.reshape(np.abs(stcl.dot(np.reshape(A[x, :, :].imag, N*N))), (N, N))

    N1 = 39
    N2 = 99
    colcarp = np.zeros((N1, N2))
    colcarp[:, :] = -100
    colcarp2 = np.zeros((N-2, N-2))
    colcarp2[:, :] = 0.01

    cmap = cm.gist_rainbow

    # create bins for the histograms
    Amps = np.arange(0.4, 1.3, 0.9/nbins)
    Laps = np.arange(0, Dmax, Dmax/100)

    # create colormap
    fc = np.zeros((N1, N2))
    for x in range(0, N1):
        for y in range(0, N2):
            c1 = np.angle((Amps[x]-np.mean(Amps))/np.max(Amps-np.mean(Amps)) +
                          1.0j*((Laps[y]-np.mean(Laps))/np.max(Laps-np.mean(Laps))))+np.pi
            fc[x, y] = c1/(2*np.pi)
    fc = cmap(fc[:, :])

    X12, Y12 = np.meshgrid(xx[1:-1:2], xx[1:-1:2])

    for x in range(0, T):
        # create colors for the right plot
        fc2 = np.zeros((N-2, N-2))
        fc2 = np.angle((np.abs(A[x, 1:-1, 1:-1])-np.mean(Amps))/np.max(Amps-np.mean(Amps)) +
                       1.0j*(np.abs(AS[x, 1:-1, 1:-1])-np.mean(Laps))/np.max(Laps-np.mean(Laps)))+np.pi
        fc2 = fc2/(2*np.pi)
        fc2 = cmap(fc2[:, :])

        # calculate histograms
        H, xedges, yedges = np.histogram2d(np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N-2)*(N-2)), np.reshape(np.abs(
            AS[x, 1:-1, 1:-1]), (N-2)*(N-2)), bins=[np.arange(0.4, 1.3, 0.9/nbins), np.arange(0, Dmax, Dmax/100)])
        H[np.where(H > 200)] = 200

        with plt.rc_context(fname='fun/matplotlibrc_3'):
            fig = plt.figure(figsize=(14, 5))
            # ,title='$t=$'+str(tt[x])
            ax = fig.add_subplot(121, projection='3d')
            X2, Y2 = np.meshgrid(yedges, xedges)
            # ax.plot_wireframe(X2[:-1,:-1],Y2[:-1,:-1],H,alpha=0.5)
            ax.plot_surface(X2[:-1, :-1], Y2[:-1, :-1], H, rstride=1, cstride=1, alpha=0.2,
                            edgecolor='k')
            surf = ax.plot_surface(X2[:-1, :-1], Y2[:-1, :-1], colcarp, rstride=1,
                                   cstride=1, norm=1, facecolors=fc, alpha=1, shade=False,
                                   rasterized=True)
            ax.set_xlim((0, Dmax))
            ax.set_ylim((0.4, 1.3))
            ax.set_zlim((-100, 200))
            ax.set_xlabel(r'$|D|$', fontsize=20)
            ax.set_ylabel(r'$A$', fontsize=20)
            ax.set_zticklabels([])
            ax.set_zlabel(r'$g$', fontsize=20)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(X[1:-1:2, 1:-1:2], Y[1:-1:2, 1:-1:2],
                             np.abs(A[x, 1:-1:2, 1:-1:2]), rstride=1, cstride=1, alpha=0.2,
                             edgecolor='k')
            surf2 = ax2.plot_surface(X12[:, :], Y12[:, :], colcarp2[::2, ::2], rstride=1, cstride=1,
                                     norm=1, facecolors=fc2[::2, ::2], alpha=1, shade=False,
                                     rasterized=True)
            ax2.set_zlim((0, 2))
            ax2.set_xlim((0, 400))
            ax2.set_ylim((0, 400))
            ax2.set_xlabel(r'$x$', fontsize=20)
            ax2.set_ylabel(r'$y$', fontsize=20)
            ax2.set_zlabel(r'$A$', fontsize=20)
            plt.savefig('fig/pdfdd_'+str(int(10000+x))+'.png')
            plt.close('all')


def main_phase():
    """Apply discrete Laplacian on the data and create histogram of its phase."""
    try:
        A = np.load('data/mcgle_data.npy')
    except:
        print('Simulating MCGLE.')
        Ad = mint.integrate(bc='periodic')
        A = Ad["data"]
        np.save('data/mcgle_data.npy', A)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(np.abs(A[-1]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('')
    plt.show()

    (T, N, N) = A.shape
    tt = np.linspace(20000, 21000, T)
    xx = np.linspace(0, 400, N)
    X, Y = np.meshgrid(xx, xx)

    stcl = stl.create_stencil(N, N, 1)

    # set parameters
    nbins = 40
    Dmax = 0.15

    AS = np.zeros((T, N, N), dtype='complex')
    for x in range(0, T):
        # AS[x, :, :] = np.reshape(np.abs(stcl.dot(np.reshape(A[x, :, :].real, N*N))), (N, N)) + \
        #     1.0j*np.reshape(np.abs(stcl.dot(np.reshape(A[x, :, :].imag, N*N))), (N, N))
        AS[x, :, :] = np.reshape(stcl.dot(np.reshape(A[x, :, :], N*N)), (N, N))

    N1 = 39
    N2 = 99
    colcarp = np.zeros((N1, N2))
    colcarp[:, :] = -100
    colcarp2 = np.zeros((N-2, N-2))
    colcarp2[:, :] = 0.01

    cmap = cm.gist_rainbow

    # create bins for the histograms
    Amps = np.linspace(0.4, 1.3, nbins)
    Laps = np.linspace(-np.pi, np.pi, 100)
    # Laps = np.linspace(-1, 1, 100)

    # create colormap
    fc = np.zeros((N1, N2))
    for x in range(0, N1):
        for y in range(0, N2):
            c1 = np.angle((Amps[x]-np.mean(Amps))/np.max(Amps-np.mean(Amps)) +
                          1.0j*((Laps[y]-np.mean(Laps))/np.max(Laps-np.mean(Laps))))+np.pi
            fc[x, y] = c1/(2*np.pi)
    fc = cmap(fc[:, :])

    X12, Y12 = np.meshgrid(xx[1:-1:2], xx[1:-1:2])

    for x in range(0, T):
        # create colors for the right plot
        fc2 = np.zeros((N-2, N-2))
        fc2 = np.angle((np.abs(A[x, 1:-1, 1:-1])-np.mean(Amps))/np.max(Amps-np.mean(Amps)) +
                       1.0j*(np.angle(AS[x, 1:-1, 1:-1])-np.mean(Laps))/np.max(Laps-np.mean(Laps)))+np.pi
        fc2 = fc2/(2*np.pi)
        fc2 = cmap(fc2[:, :])

        # calculate histograms
        H, xedges, yedges = np.histogram2d(np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N-2)*(N-2)), np.reshape(np.angle(
            AS[x, 1:-1, 1:-1]), (N-2)*(N-2)), bins=[Amps, Laps])
        H[np.where(H > 200)] = 200

        with plt.rc_context(fname='fun/matplotlibrc_3'):
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
            X2, Y2 = np.meshgrid(yedges, xedges)
            ax.plot_surface(X2[: -1, : -1], Y2[: -1, : -1], H, rstride=1, cstride=1, alpha=0.2,
                            edgecolor='k')
            surf = ax.plot_surface(X2[:-1, :-1], Y2[:-1, :-1], colcarp, rstride=1,
                                   cstride=1, norm=1, facecolors=fc, alpha=1, shade=False,
                                   rasterized=True)
            ax.set_xlim((Laps[0], Laps[-1]))
            ax.set_ylim((0.4, 1.3))
            ax.set_zlim((-100, 200))
            ax.set_xlabel(r'$\angle D$', fontsize=20)
            ax.set_ylabel(r'$A$', fontsize=20)
            ax.set_zticklabels([])
            ax.set_zlabel(r'$g$', fontsize=20)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(X[1:-1:2, 1:-1:2], Y[1:-1:2, 1:-1:2],
                             np.abs(A[x, 1:-1:2, 1:-1:2]), rstride=1, cstride=1, alpha=0.2,
                             edgecolor='k')
            surf2 = ax2.plot_surface(X12[:, :], Y12[:, :], colcarp2[::2, ::2], rstride=1, cstride=1,
                                     norm=1, facecolors=fc2[::2, ::2], alpha=1, shade=False,
                                     rasterized=True)
            ax2.set_zlim((0, 2))
            ax2.set_xlim((0, 400))
            ax2.set_ylim((0, 400))
            ax2.set_xlabel(r'$x$', fontsize=20)
            ax2.set_ylabel(r'$y$', fontsize=20)
            ax2.set_zlabel(r'$A$', fontsize=20)
            plt.savefig('fig/pdfphase_'+str(int(10000+x))+'.png')
            plt.close('all')


if __name__ == "__main__":
    main_amp()
    main_phase()
