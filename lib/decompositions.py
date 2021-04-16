import numpy as np
from lib.linear_system import gauss_elimination
from lib.matrix import Matrix

def QR_method_gram_sh(mtx):
    Q = mtx.gram_shmidt()
    m = mtx.data.shape[0]
    n = mtx.data.shape[1]
    R = np.zeros([n, n])
    
    for i in range(0, n):
        A = np.concatenate((Q, np.array([mtx.data[:, i]]).T), axis=1)
        r = gauss_elimination(Matrix(A))
        R[:, i] = r[0]
        
    return Q, R

def QR_method_hh(mtx):
    R, Q = mtx.house_holder()
    
    return Q, R

def eign_qr_hh(mtx):
    Q, R = QR_method_hh(mtx)
    
    eignval = np.diag(R)
    D = np.zeros([mtx.data.shape[0], eignval.shape[0]])
    for i in range(eignval.shape[0]):
        A = mtx.data - eignval[i] * np.eye(mtx.data.shape[0])
        b = np.zeros([1, A.shape[0]]).T
        b[A.shape[0] - 1, 0] = 1 # chute para sair da solução trivial
        A = np.concatenate((A, b), axis=1)
        
        x = gauss_elimination(Matrix(A))
        x = x / np.linalg.norm(x)   
        D[:, i] = x
        
    return Q, eignval

def svd(mtx):
    A = mtx.data
    m = A.shape[0]
    n = A.shape[1]

    M = np.matmul(A, A.T)
    U, eig_M = eign_qr_hh(Matrix(M))

    S = np.eye(m)
    for i in range(0, m):
        S[i, i] = np.sqrt(np.abs(eig_M[i]))
    
    V = np.matmul(np.linalg.inv(S), U.T)
    V = np.matmul(V, A)

    return U, S, V