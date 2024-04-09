import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from int.mcgle_2d import integrate
import stencil as stl

results_dir = "data/mcgle_2d/"

data = integrate(tmin=300, tmax=400)
A = data["data"]
(T,N,N) = A.shape
A = np.reshape(A,(T,N*N))
tt = np.arange(1500,1600,100.0/T)
xx = np.arange(0,400,400.0/N)
X, Y = np.meshgrid(xx, xx)

A_mean = np.mean(A, axis=1)
A_rot = (A[:, :]/A_mean[:, None])-1
A = A_rot

nbins=101

ar = 1.3

Areal = np.arange(-ar,ar,2*ar/nbins)

stcl = stl.create_stencil(N,N,1)

Dmax = 0.15

AS = np.zeros((T,N,N),dtype='complex')
for x in range(0,T):
    AS[x,:,:] = np.reshape(stcl.dot(A[x,:]),(N,N))

ASreal = np.arange(-Dmax,Dmax,2*Dmax/nbins)
Aabs = np.arange(0,ar,ar/nbins)

A = np.reshape(A,(T,N,N))

nbins=51

ASreal = np.arange(-Dmax,Dmax,2*Dmax/nbins)
Aabs = np.arange(0,ar,ar/nbins)

for x in range(0, T):
    H, edges = np.histogramdd(np.vstack((np.reshape(AS[x, 1:-1, 1:-1].real, (N - 2)**2), np.reshape(
        AS[x, 1:-1, 1:-1].imag, (N - 2)**2), np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N - 2)**2))).T, bins=[ASreal, ASreal, Aabs])
    H1, xedges, yedges = np.histogram2d(np.reshape(
        AS[x, 1:-1, 1:-1].real, (N - 2)**2), np.reshape(AS[x, 1:-1, 1:-1].imag, (N - 2)**2), bins=[ASreal, ASreal])
    H1 = H1 + 1
    H1 = np.log(H1)
    H2, yedges2, zedges2 = np.histogram2d(np.reshape(
        AS[x, 1:-1, 1:-1].real, (N - 2)**2), np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N - 2)**2), bins=[ASreal, Aabs])
    H2 = H2 + 1
    H2 = np.log(H2)
    H3, xedges3, zedges3 = np.histogram2d(np.reshape(
        AS[x, 1:-1, 1:-1].imag, (N - 2)**2), np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N - 2)**2), bins=[ASreal, Aabs])
    H3 = H3 + 1
    H3 = np.log(H3)
    H = H + 1
    H = np.log(H)
    with plt.rc_context(fname='matplotlibrc_3'):
        fig = plt.figure(figsize=(7, 5), facecolor='1', edgecolor='1')
        X, Y, Z = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1])
        X1, Y1 = np.meshgrid(xedges[:-1], yedges[:-1])
        Y2, Z2 = np.meshgrid(yedges2[:-1], zedges2[:-1])
        X3, Z3 = np.meshgrid(xedges3[:-1], zedges3[:-1])
        ax = fig.add_subplot(111, projection='3d')
        scat1 = ax.scatter(np.reshape(X, (nbins - 1)**3), np.reshape(Y, (nbins - 1)**3), np.reshape(Z, (nbins - 1)**3),
                           s=np.reshape(H, (nbins - 1)**3), lw=0, c=cm.jet(np.reshape(H / np.max(H), (nbins - 1)**3)), rasterized=True)
        ax.set_xlabel(r'Re $ D$', fontsize=20)
        ax.set_ylabel(r'Im $ D$', fontsize=20)
        ax.set_zlabel(r'$|W|$', fontsize=20)
        ax.view_init(azim=-50, elev=-170)
        ax.patch.set_visible(False)
        fig.patch.set_facecolor('white')
        plt.savefig(results_dir + "pdf_3d_%04d" % x + '.png')
        plt.close('all')
