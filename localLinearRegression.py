import numpy as np
import numpy.linalg as lg

# vecs: The matrix with the DMAP vectors in the columns
def gaussian_kde(vecs, eps_reg):
    reg_kernel = lambda x, y: np.exp(-np.real(np.vdot(x - y, x - y)) / eps_reg**2)
    N = vecs.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            K[i, j] = reg_kernel(vecs[i,:], vecs[j,:])
            K[j, i] = K[i, j]

    return K
    
# vecs: The matrix with the DMAP vectors in the columns
def localLinearRegression(vecs, x, eps_reg):
    K = gaussian_kde(vecs, eps_reg)
    N = vecs.shape[0]
    L = vecs.shape[1]

    alphas = np.zeros(N, dtype=np.complex64)
    betas = np.zeros((L, N), dtype=np.complex64)
    for i in range(N):
        R = np.zeros((L+1, L+1), dtype=np.complex64)
        R[0,0] = 2.0 * np.sum(K[i,:])
        R[0,1:] = 2.0 * np.sum(K[i,:][:,np.newaxis] * vecs, axis=0)
        R[1:,0] = R[0,1:] # Symmetric for some reason
        R[1:,1:] = 2.0 * np.sum(np.array([K[i, j] * np.outer(vecs[j,:], vecs[j,:]) for j in range(N)]))

        rhs = np.zeros(L+1, dtype=np.complex64)
        rhs[0] = 2.0 * np.sum(K[i,:] * x)
        rhs[1:] = 2.0 * np.sum( (K[i,:] * x)[:, np.newaxis] * vecs)

        p = lg.solve(R, rhs)
        alphas[i] = p[0]
        betas[:,i] = p[1:]

    # Calculate the r-value
    diff_vector = x - (alphas + np.dot(vecs, betas))
    r = np.sqrt(np.real(np.vdot(diff_vector, diff_vector)) / np.real(np.vdot(x, x)))

    return r