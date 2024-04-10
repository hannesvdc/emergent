import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from int.sle_lin import integrate

results_dir = "data/sle/"

# Create SL data.
data = integrate(n_oscillators=2**12, t_min=1000, t_max=1100)

tt = np.linspace(data["t_min"], data["t_max"], data["n_time_steps"])
xx = np.arange(0, data["n_oscillators"])

data_mean = np.mean(data["data"], axis=1)
data_rot = (data["data"][:, :] / data_mean[:, None]) - 1

nbins = 100

# Set boundaries for 2d histogram (independent of timestep):
min_real, max_real = (np.min(data_rot.real), np.max(data_rot.real))
min_imag, max_imag = (np.min(data_rot.imag), np.max(data_rot.imag))

min_max_real = (min_real * 1.05 - max_real * 0.05, max_real * 1.05 - min_real * 0.05)
min_max_imag = (min_imag * 1.05 - max_imag * 0.05, max_imag * 1.05 - min_imag * 0.05)

Areal = np.linspace(min_max_real[0], min_max_real[1], nbins + 1)
Aimag = np.linspace(min_max_imag[0], min_max_imag[1], nbins + 1)

# Create 2d grids for the histogram plot:
X2, Y2 = np.meshgrid(Areal, Aimag)

for t in range(data["n_time_steps"]): #+1?
    print(t)

    # Create histogram of oscillator positions in the complex plane.
    H, yedges, xedges = np.histogram2d(data_rot[t].imag, data_rot[t].real, bins=[Aimag, Areal])

    cmap = cm.gist_rainbow

    # Color all grid points within the span of the histogram according to relative position.
    relative_x = np.matrix((Areal[:] - np.mean(Areal)) / np.max(np.abs(Areal[:] - np.mean(Areal))))
    relative_y = np.matrix((Aimag[:] - np.mean(Aimag)) /
                           np.max(np.abs(Aimag[:] - np.mean(Aimag)))).T

    fc = (np.angle(np.array(relative_x - 1j * relative_y)) + np.pi) / (2 * np.pi)
    fc = cmap(fc)

    # Do the same for the values actually occupied by individual oscillators.
    relative_Ax = (data_rot[t].real - np.mean(Areal)) / np.max(np.abs(data_rot[t].real - np.mean(Areal)))
    relative_Ay = (data_rot[t].imag - np.mean(Aimag)) / np.max(np.abs(data_rot[t].imag - np.mean(Aimag)))

    fc2 = (np.angle(np.array(relative_Ax - 1j * relative_Ay)) + np.pi) / (2 * np.pi)
    fc2 = cmap(fc2)

    elev = 25
    azim = -155

    H[np.where(H > 200)] = 200
    fig = plt.figure(1, figsize=(15, 7))

    ax = fig.add_subplot(121, projection='3d')  # ,axisbg=(0.0, 0.0, 0.0))
    ax.plot_surface(X2[:-1, :-1], Y2[:-1, :-1], H, rstride=1, cstride=1,
                    norm=1, facecolors=fc, alpha=1, shade=False)
    ax.plot_wireframe(X2[:-1, :-1], Y2[:-1, :-1], H, color="black", alpha=0.25)
    ax.set_xlim(min_max_real)
    ax.set_ylim(min_max_imag)

    ax.view_init(elev, azim)
    ax.patch.set_visible(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel(r'$\mathrm{Re}(W)$', fontsize=20, labelpad=10)
    ax.set_ylabel(r'$\mathrm{Im}(W)$', fontsize=20, labelpad=10)
    ax.set_zlabel(r'$g$', fontsize=30)

    ax2 = fig.add_subplot(122)
    plt.title('$t=$' + "%04.2f" % tt[t])
    ax2.scatter(np.arange(len(data_rot[t])), np.real(data_rot[t]), c=fc2, lw=0)
    ax2.set_xlim((0, data["n_oscillators"]))
    ax2.set_ylim(min_max_real)
    ax2.set_xlabel(r'$x$', fontsize=20)
    ax2.set_ylabel(r'$\mathrm{Re}(W)$', fontsize=20)

    plt.savefig(results_dir + "sl_surf_%04d" % t + '.png')
    plt.clf()
