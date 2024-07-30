import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import int.mcgle_2d as mint
import fun.stencil_2d as stl


def main_pdf3d():
    try:
        A = np.load('data/mcgle_data.npy')
    except:
        print('Simulating MCGLE.')
        Ad = mint.integrate(L=400, N=512, tmin=2000, tmax=2500)
        A = Ad["data"]
        np.save('data/mcgle_data.npy', A)
    # A = np.load('/home/felix/paper_yannis/MCGLE2D_data_0.2_0.61_1.5_1.0_1600.npy')
    (T, N, N) = A.shape
    A = np.reshape(A, (T, N*N))
    tt = np.linspace(1500, 1600, 1000)
    xx = np.linspace(0, 400, N)
    X, Y = np.meshgrid(xx, xx)

    A_mean = np.mean(A, axis=1)
    # A_rot = (A[:, :]/A_mean[:, None])-1
    A_rot = (A[:, :]/A_mean[:, None])
    A = A_rot

    nbins = 101

    ar = 1.3

    Areal = np.linspace(-ar, ar, nbins)

    stcl = stl.create_stencil(N, N, 1)

    Dmax = 0.15

    AS = np.zeros((T, N, N), dtype='complex')
    for x in range(0, T):
        AS[x, :, :] = np.reshape(stcl.dot(A[x, :]), (N, N))

    ASreal = np.arange(-Dmax, Dmax, 2*Dmax/nbins)
    Aabs = np.arange(0, ar, ar/nbins)

    A = np.reshape(A, (T, N, N))

    nbins = 51

    ASreal = np.arange(-Dmax, Dmax, 2*Dmax/nbins)
    Aabs = np.arange(0, ar, ar/nbins)

    x = 0

    for x in range(0, T):
        H, edges = np.histogramdd(np.vstack((np.reshape(AS[x, 1:-1, 1:-1].real, (N-2)**2), np.reshape(
            AS[x, 1:-1, 1:-1].imag, (N-2)**2), np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N-2)**2))).T, bins=[ASreal, ASreal, Aabs])
        H1, xedges, yedges = np.histogram2d(np.reshape(
            AS[x, 1:-1, 1:-1].real, (N-2)**2), np.reshape(AS[x, 1:-1, 1:-1].imag, (N-2)**2), bins=[ASreal, ASreal])
        H1 = H1+1
        H1 = np.log(H1)
        H2, yedges2, zedges2 = np.histogram2d(np.reshape(
            AS[x, 1:-1, 1:-1].real, (N-2)**2), np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N-2)**2), bins=[ASreal, Aabs])
        H2 = H2+1
        H2 = np.log(H2)
        H3, xedges3, zedges3 = np.histogram2d(np.reshape(
            AS[x, 1:-1, 1:-1].imag, (N-2)**2), np.reshape(np.abs(A[x, 1:-1, 1:-1]), (N-2)**2), bins=[ASreal, Aabs])
        H3 = H3+1
        H3 = np.log(H3)
        H = H+1
        H = np.log(H)
        with plt.rc_context(fname='matplotlibrc_3'):
            fig = plt.figure(figsize=(7, 5), facecolor='1', edgecolor='1')
            # ,title='$t=$'+str(tt[x])
            X, Y, Z = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1])
            X1, Y1 = np.meshgrid(xedges[:-1], yedges[:-1])
            Y2, Z2 = np.meshgrid(yedges2[:-1], zedges2[:-1])
            X3, Z3 = np.meshgrid(xedges3[:-1], zedges3[:-1])
            ax = fig.add_subplot(111, projection='3d', facecolor='w')
            scat1 = ax.scatter(np.reshape(X, (nbins-1)**3), np.reshape(Y, (nbins-1)**3), np.reshape(Z, (nbins-1)**3),
                               s=np.reshape(H, (nbins-1)**3), lw=0, c=cm.jet(np.reshape(H/np.max(H), (nbins-1)**3)), rasterized=True)
            scat2 = ax.scatter(np.reshape(X1, (nbins-1)**2), np.reshape(Y1, (nbins-1)**2), np.zeros(X1.shape), s=np.reshape(
                H1, (nbins-1)**2), lw=0, c=cm.Greys(np.reshape(H1/np.max(H1), (nbins-1)**2)), rasterized=True)
            scat3 = ax.scatter(np.zeros(Y2.shape)-Dmax, np.reshape(Y2, (nbins-1)**2), np.reshape(Z2, (nbins-1)**2), s=np.reshape(
                H2.T, (nbins-1)**2), lw=0, c=cm.Greys(np.reshape(H2.T/np.max(H2), (nbins-1)**2)), rasterized=True)
            scat4 = ax.scatter(np.reshape(X3, (nbins-1)**2), np.zeros(X3.shape)+Dmax, np.reshape(Z3, (nbins-1)**2), s=np.reshape(
                H3.T, (nbins-1)**2), lw=0, c=cm.Greys(np.reshape(H3.T/np.max(H3), (nbins-1)**2)), rasterized=True)
            ax.set_xlabel(r'Re $D$', fontsize=20)
            ax.set_ylabel(r'Im $D$', fontsize=20)
            ax.set_zlabel(r'$|W|$', fontsize=20)
            ax.patch.set_visible(False)
            fig.patch.set_facecolor('white')
            plt.savefig('fig/pdf3d_proj_'+str(int(10000+x))+'.png')
            plt.close('all')
    # cmd = 'ffmpeg -framerate 18 -start_number 10000 -i /home/felix/paper_yannis/fig/pdf_proj_%d.png -bit_rate 512kbps -frames 1000 -vframes 1000 -r 18 -vcodec mpeg4 -qscale:v 8 /home/felix/paper_yannis/pdf_3d_proj.avi'
    # os.system(cmd)
