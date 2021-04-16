import numpy as np

# eliminação de gauss com pivotação parcial e matriz escalonada

def gauss_elimination(M):
    A = M.partial_pivot_echelon()
    s = np.zeros(M.data.shape[1] - 1)

    A = A.data
    m = A.shape[0]
    n = A.shape[1]

    d = n - 2
    for i in range(m - 1, -1, -1):
        res = A[i][n - 1]

        if d < n - 2:
            for j in range(d + 1, n - 1):
                res = res - A[i][j] * s[j]
            
        if A[i][d] != 0:
            res = res / A[i][d]
        else:
            res = 0
        
        s[d] = res
        d = d - 1
    
    return np.array([s])

#eliminação de gauss com pivotação parcial e matriz escalonada reduzida

def gauss_elimination_reduction(M):
    A = M.partial_pivot_echelon_reduced()
    s = np.zeros(M.data.shape[1] - 1)

    A = A.data
    m = A.shape[0]
    n = A.shape[1]

    d = n - 2
    for i in range(m - 1, -1, -1):
        res = A[i][n - 1]

        if d < n - 2:
            for j in range(d + 1, n - 1):
                res = res - A[i][j] * s[j]
            
        if A[i][d] != 0:
            res = res / A[i][d]
        else:
            res = 0
        
        s[d] = res
        d = d - 1
    
    return np.array([s])