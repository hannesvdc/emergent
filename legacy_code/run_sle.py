import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import int.sle as sle
import int.sle_lin as slin


def main_sle():
    """Create historgams of oscillators."""
    try:
        A = np.load('data/sle_data.npy')
    except:
        print('Simulating Stuart-Landau ensemble.')
        Ad = slin.integrate(c2=2.0, kre=0.7, kim=-0.7, N=2048,
                            tmin=2000, tmax=2500, ic='weakrandom')
        A = Ad["data"]
        np.save('data/sle_data.npy', A)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.abs(A[-1]), '.')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.savefig('')
        plt.show()

    (T, N) = A.shape

    for i in range(T):
        A[i] = A[i]*np.exp(-1.0j*np.angle(np.mean(A[i])))

    for i in range(T):
        A[i] = A[i] - np.mean(A[i, -1])

    A = -A

    Amps = np.linspace(-1.1, 0.1, 100)
    Laps = np.linspace(-0.6, 0.6, 100)

    cmap = cm.gist_rainbow

    # create colormap
    N1 = 99
    N2 = 99
    fc = np.zeros((N1, N2))
    for x in range(0, N1):
        for y in range(0, N2):
            c1 = np.angle((Amps[x]-np.mean(Amps))/np.max(Amps-np.mean(Amps)) +
                          1.0j*((Laps[y]-np.mean(Laps))/np.max(Laps-np.mean(Laps))))+np.pi
            fc[x, y] = c1/(2*np.pi)
    fc = cmap(fc[:, :])
    colcarp = np.zeros((N1, N2))
    colcarp[:, :] = -0.05

    for x in range(0, T):
        # create colors for the right plot
        fc2 = np.zeros((N-2, N-2))
        fc2 = np.angle((np.real(A[x])-np.mean(Amps))/np.max(Amps-np.mean(Amps)) +
                       1.0j*(np.imag(A[x])-np.mean(Laps))/np.max(Laps-np.mean(Laps)))+np.pi
        fc2 = fc2/(2*np.pi)
        fc2 = cmap(fc2)

        # calculate histograms
        H, xedges, yedges = np.histogram2d(np.real(A[x]), np.imag(A[x]), bins=[Amps, Laps])
        H[np.where(H > 200)] = 200

        with plt.rc_context(fname='fun/matplotlibrc_3'):
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
            X2, Y2 = np.meshgrid(yedges, xedges)
            ax.plot_surface(X2[: -1, : -1], Y2[: -1, : -1], H, rstride=2, cstride=2, alpha=0.15,
                            edgecolor='k')
            surf = ax.plot_surface(X2[:-1, :-1], Y2[:-1, :-1], colcarp, rstride=1,
                                   cstride=1, norm=1, facecolors=fc, alpha=1, shade=False,
                                   rasterized=True)
            ax.set_zlim((-0.5, 200))
            ax.set_ylim((Amps[0], Amps[-1]))
            ax.set_xlim((Laps[0], Laps[-1]))
            ax.set_xlabel(r'Im $W$', fontsize=20)
            ax.set_ylabel(r'Re $W$', fontsize=20)
            ax.set_zticklabels([])
            ax.set_zlabel(r'$g$', fontsize=20)
            ax2 = fig.add_subplot(122)
            ax2.scatter(np.arange(N), np.real(A[x]), color=fc2)
            ax2.set_ylim((Amps[0], Amps[-1]))
            ax2.set_xlabel(r'$i$', fontsize=20)
            ax2.set_ylabel(r'Re $W$', fontsize=20)
            plt.savefig('fig/sle_'+str(int(10000+x))+'.png')
            plt.close('all')


if __name__ == "__main__":
    main_sle()
