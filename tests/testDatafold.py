import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import sklearn.manifold as manifold
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA

import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector

# Generate S-curved point cloud
nr_samples = 15000

# reduce number of points for plotting
nr_samples_plot = 10000
idx_plot = np.random.permutation(nr_samples)[0:nr_samples_plot]

# generate point cloud
X, X_color = make_swiss_roll(nr_samples, random_state=3, noise=0)

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X[idx_plot, 0],
    X[idx_plot, 1],
    X[idx_plot, 2],
    c=X_color[idx_plot],
    cmap=plt.cm.Spectral,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("point cloud on S-shaped manifold")
plt.show()
plt.close()

# Optimize kernel parameters
X_pcm = pfold.PCManifold(X)
X_pcm.optimize_parameters(n_subsample =5000,tol =1e-08,k= 100)
print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

# Fit DiffusionMap model
dmap = dfold.DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
    n_eigenpairs=10
)
dmap = dmap.fit(X_pcm)
evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_
fig = plt.figure(figsize=(3*2,3*4))
for i in range (1,9):
    ax = fig.add_subplot(4,2,i)
    ax.scatter(evecs[:,1],evecs[:,i+1],s=0.1,c='b')
    ax.set_xlabel(r'$\phi_1$',fontsize=15)
    ax.set_ylabel(r"$\phi_{{{}}}$".format(i+1),fontsize=15)
plt.show()
plt.close()
print(dmap.eigenvectors_)

# Remove unnecessary or harmonic eigenvectors
selection = LocalRegressionSelection(
    intrinsic_dim=2, n_subsample=500, strategy="dim"
).fit(dmap.eigenvectors_)
print(f"Found parsimonious eigenvectors (indices): {selection.evec_indices_}")
print(selection.residuals_)
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot(selection.residuals_,'o:k')
ax.set_xlabel(r'$\lambda_i$',fontsize=15)
ax.set_ylabel(r'Residual',fontsize=15)
plt.show()

target_mapping = selection.transform(dmap.eigenvectors_)

f, ax = plt.subplots(figsize=(4,4))
ax.scatter(
    target_mapping[idx_plot, 0],
    target_mapping[idx_plot, 1],
    c=X[idx_plot,1],
    cmap=plt.cm.Spectral,
)
ax.set_xlabel(r'$\phi_1$',fontsize=15)
ax.set_ylabel(r'$\phi_5$',fontsize=15)
plt.show()