import numpy as np
import scipy.linalg as slg
import matplotlib.pyplot as plt

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

def computeDiffusionMap(eps):
    chunk_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/emergent/dmaps/'
    filename = 'timechunks.npy'
    X = np.load(chunk_directory + filename)
    M = 5000
    X = X[:, np.random.permutation(X.shape[1])] [:, 0:M] # subsample for speed and memory usage

    print('Constructing kernel matrix')
    kernel = lambda x: np.exp(-np.sum(np.abs(x[:,np.newaxis] - X)**2, axis=0) / eps)
    K = np.zeros((X.shape[1], X.shape[1]))
    for n in range(X.shape[1]):
        K[n, :] = kernel(X[:,n])
    P = K / np.sum(K, axis=1, keepdims=True)

    # Compute eigenvalues to see a cut-off
    print('Computing Eigenvalues')
    eigvals, eigvecs = slg.eig(P, left=True, right=False)
    for n in range(20):
        print(n+1, np.vdot(eigvecs[:,0], eigvecs[:,n+1]))
    plt.plot(np.arange(len(eigvals)), np.real(eigvals), marker='o', label='Left Eigenvalues in Descending Order')
    plt.legend()
    plt.figure()
    plt.scatter(eigvecs[:,0]/slg.norm(eigvecs[:,0]), eigvecs[:,10]/slg.norm(eigvecs[:,10]))
    plt.xlabel(r'$\psi_0$')
    plt.ylabel(r'$\psi_3$', rotation=0)
    plt.show()


if __name__ == '__main__':
    eps = 10.0
    computeDiffusionMap(eps)