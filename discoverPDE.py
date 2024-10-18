import numpy as np
import scipy.linalg as slg
import matplotlib.pyplot as plt

import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector

from localLinearRegression import *

def createGLDataset():
    data_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/emergent/Ginzburg_Landau_3D/'
    params = {'c1': 0.2, 'c2': 0.61, 'nu': 1.5, 'eta': 1.0}
    T0 = 4500.0
    Dt = 0.1
    seed = 100
    parameters_in_filename = lambda temp: '_c1='  + str(params['c1']) \
                                        + '_c2='  + str(params['c2']) \
                                        + '_nu='  + str(params['nu']) \
                                        + '_eta=' + str(params['eta']) \
                                        + '_seed='+ str(seed) \
                                        + '_T='   + str(temp)+'.npy'
    
    # Load the data and store in a giant tensor with millions of entries.
    dataset = np.zeros((100, 512, 512), dtype=np.complex64)
    for n in range(100):
        t = T0 + n * Dt
        print('t =', t)
        filename = 'Ginzburg_Landau_ETD2_Evolution' + parameters_in_filename(t)
        data = np.load(data_directory + filename)
        dataset[n,:,:] = data
    print(dataset.shape, np.count_nonzero(dataset == 0.0 + 1j*0.0))

    # Slice the data into time chunks.
    chunk_data = np.empty((10,0), dtype=np.complex64)
    for n in range(10):
        chunk_data = np.append(chunk_data, np.reshape(dataset[10*n:10*(n+1), :, :], newshape=(10, 512**2)), axis=1)
    
    # Shuffle the chunk columns and store them.
    chunks = chunk_data[:, np.random.permutation(chunk_data.shape[1])]
    chunk_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/emergent/dmaps/'
    filename = 'timechunks'
    np.save(chunk_directory + filename, chunks)

def computeDiffusionMapDatafold():
    chunk_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/emergent/dmaps/'
    filename = 'timechunks.npy'
    data = np.load(chunk_directory + filename)
    M = 5000
    plot_idx = np.random.permutation(data.shape[1])
    data = data[:, plot_idx] [:, 0:M] # subsample for speed and memory usage
    X = np.concatenate((np.real(data.T), np.imag(data.T)), axis=1)
    print('data shape', X.shape)

    X_pcm = pfold.PCManifold(X)
    print(type(X_pcm))
    X_pcm.optimize_parameters(n_subsample =5000,tol =1e-08,k= 100)
    print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

    # Fit DiffusionMap model
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=100.0 * X_pcm.kernel.epsilon),
        n_eigenpairs=100
    )
    dmap = dmap.fit(X_pcm)

    # Remove unnecessary or harmonic eigenvectors
    selection = LocalRegressionSelection(
        intrinsic_dim=10, n_subsample=500, strategy="dim"
    ).fit(dmap.eigenvectors_)
    print(f"Found parsimonious eigenvectors (indices): {selection.evec_indices_}")
    print(selection.residuals_)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.plot(selection.residuals_,'o:k')
    ax.set_xlabel(r'$\lambda_i$', fontsize=15)
    ax.set_ylabel(r'Residual', fontsize=15)
    ax.set_title(r"$\varepsilon = $" + " {eps:.4f}".format(eps=100.0 * X_pcm.kernel.epsilon))
    plt.show()

    target_mapping = selection.transform(dmap.eigenvectors_)
    print(type(target_mapping), target_mapping.shape)

    # using the variable axs for multiple Axes
    fig, axs = plt.subplots(2, 5)
    fig.suptitle(r"$\varepsilon = $" + " {eps:.4f}".format(eps=100.0 * X_pcm.kernel.epsilon))
    for ax_index in range(10):
        row_index = 1 + ax_index // 5
        col_index = 1 + ax_index % 5

        ax = axs[row_index-1, col_index-1]
        ax.scatter(
            target_mapping[:, ax_index],
            target_mapping[:, 0],
            c=X[:,ax_index],
            cmap=plt.cm.Spectral,
        )
        ax.set_xlabel("$" + "\phi_{" + "{n}".format(n=ax_index+1) + "}$",fontsize=15)
        if col_index == 1:
            ax.set_ylabel(r'$\phi_1$',fontsize=15)
    plt.show()

def computeDiffusionMap(eps):
    chunk_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/emergent/dmaps/'
    filename = 'timechunks.npy'
    X = np.load(chunk_directory + filename)
    M = 5000
    X = X[:, np.random.permutation(X.shape[1])] [:, 0:M] # subsample for speed and memory usage

    print('Constructing kernel matrix')
    kernel = lambda x: np.exp(-np.sum(np.abs(x[:,np.newaxis] - X)**2, axis=0) / eps)
    W = np.zeros((X.shape[1], X.shape[1]))
    for n in range(X.shape[1]):
        W[n, :] = kernel(X[:,n])
    D = 1.0 / np.sum(W, axis=1, keepdims=True)
    W_hat = D * W * D # D is a ful matrix containing all diagonal entries on the whole row.
    D_hat = np.sum(W_hat, axis=1, keepdims=True)
    P = W_hat / D_hat

    # Compute eigenvalues to see a cut-off
    print('Computing Eigenvalues')
    eigvals, eigvecs = slg.eig(P, right=True)

    # Calculate the r-values to discover a parsimonious representation
    print('Calculating R-values')
    r_values = []
    eps_reg = 1.0
    for i in range(eigvecs.shape[1] - 1):
        print('k =', i)
        r = localLinearRegression(eigvecs[:,0:i], eigvecs[i], eps_reg)
        print(r)
        r_values.append(r)

    plt.plot(np.arange(len(eigvals)), np.real(eigvals), marker='o', label='Left Eigenvalues in Descending Order')
    plt.legend()

    plt.figure()
    plt.semilogy(np.arange(len(r_values)), r_values, label='R-values')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$r$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    eps = 1.0
    #computeDiffusionMap(eps)
    computeDiffusionMapDatafold()