import numpy as np
import math

class Matrix():
    
    def __init__(self, data):
        self.data = np.array(data, dtype='float64')
    
    def transpose(self):
        return np.array(self.data).T

    def scalar_product(self, scalar):
        res = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                res[i][j] = self.data[i][j] * scalar

        return Matrix(res)
    
    def matrix_product(self, mtx):
        if(self.data.shape[1] != mtx.data.shape[0]):
            print('Erro - As dimensões das matrizes são incompatíveis para realização do produto.')
            return None
        
        m = self.data.shape[0]
        n = mtx.data.shape[1]
        k = self.data.shape[1]
        res = np.zeros([m, n])
        
        for l in range(m):
            for c in range(n):
                s = 0
                for v in range(k):
                    s = s + self.data[l][v] * mtx.data[v][c]
                
                res[l][c] = s

        return Matrix(res)
    
    # forma escalonada reduzida da matriz via pivotação parcial

    def partial_pivot_echelon_reduced(self):
        A = np.array(self.data)
        m = self.data.shape[0]
        n = self.data.shape[1]

        d = 0
        for c in range(n):
            if d >= m:
                break
            elif d == m - 1:
                pivot = A[d][d]

            
            # caso o pivô seja igual a zero, troca a linha 
            # com a próxima diferente de zero

            if d < m - 1:
                max_v = abs(A[d][c])
                idx_v = d
                for i in range(d+1, m):
                    if abs(A[i][c]) > max_v:
                        max_v = A[i][c]
                        idx_v = i
                
                pivot = A[idx_v][c]    
                for j in range(n):
                    aux = A[d][j]
                    A[d][j] = A[idx_v][j]
                    A[idx_v][j] = aux

            # zera elementos abaixo da linha do pivot

            for i in range(d+1, m):
                if A[i][c] != 0:
                    v = -A[i][c]/float(pivot)

                    for j in range(c, n):
                        A[i][j] = round(A[i][j] + v * A[d][j], 5)
            
            # zera elementos acima da linha do pivot

            if d > 0:
                for i in range(0, d):
                    v = -A[i][c]/float(pivot)

                    for j in range(c, n):
                        A[i][j] = round(A[i][j] + v * A[d][j], 5)

            # torna linha do pivot pivot igual a 1

            if pivot != 1 and pivot != 0:
                for j in range(c, n):
                    A[d][j] = round(A[d][j] / float(pivot), 5)

            d = d + 1        
        return Matrix(A)

    # forma escalonada da matriz via pivotação parcial

    def partial_pivot_echelon(self):
        A = np.array(self.data)
        m = self.data.shape[0]
        n = self.data.shape[1]

        d = 0
        for c in range(n):
            if d >= m:
                break
            elif d == m - 1:
                pivot = A[d][d]
            
            if d < m - 1:
                max_v = abs(A[d][c])
                idx_v = d
                for i in range(d+1, m):
                    if abs(A[i][c]) > max_v:
                        max_v = A[i][c]
                        idx_v = i
                
                pivot = A[idx_v][c]    
                for j in range(n):
                    aux = A[d][j]
                    A[d][j] = A[idx_v][j]
                    A[idx_v][j] = aux
            
            for i in range(d+1, m):
                if A[i][c] != 0:
                    v = -A[i][c]/float(pivot)

                    for j in range(c, n):
                        A[i][j] = round(A[i][j] + v * float(A[d][j]), 5)
                        
            d = d + 1        
        return Matrix(A)

    def LU(self):
        A = np.array(self.data)
        m = self.data.shape[0]
        n = self.data.shape[1]

        p_n = np.zeros([1, n])

        for i in range(0, n):
            p_n[0][n] = i

        for k in range(0, n):
            pivo = A[k, k]
            l_pivo = k
            for i in range(k + 1, n):
                if np.abs(A[i, k]) > np.abs(pivo):
                    pivo = A[i, k]
                    l_pivo = i
            
            if pivo == 0:
                print("Erro - Matriz não eh singular.")
                break

            if l_pivo != k:
                troca = p_n[k]
                p_n[k] = p[l_pivo]
                p[l_pivo] = troca

                for j in range(n):
                    troca = A[k, j]
                    A[k, j] = A[l_pivo, j]
                    A[l_pivo, j] = troca


                for i in range(k, n):
                    m = A[i, k] / A[k, k]
                    A[i, k] = m

                    for j in range(k, n):
                        A[i, j] = A[i, j] - m * A[k, j]

            





    # fatoração LU

    def fatoracao_lu(self):
        A = np.array(self.data)
        m = self.data.shape[0]
        n = self.data.shape[1]
        swap_lines = np.array(np.linspace(0, m - 1, m), dtype='int')

        if m != n:
            print("Para fatoração LU, a matriz precisa ser quadrada.")
            return

        L = np.eye(n)

        d = 0
        for c in range(n):
            if d >= m:
                break
            elif d == m - 1:
                pivot = A[d][d]
            
            if d < m - 1:
                max_v = abs(A[d][c])
                idx_v = d
                for i in range(d+1, m):
                    if abs(A[i][c]) > max_v:
                        max_v = A[i][c]
                        idx_v = i
                
                aux = swap_lines[idx_v]
                swap_lines[idx_v] = swap_lines[d]
                swap_lines[d] = aux

                pivot = A[idx_v][c]    
                for j in range(n):
                    aux = A[d][j]
                    A[d][j] = A[idx_v][j]
                    A[idx_v][j] = aux
            
            for i in range(d+1, m):
                if A[i][c] != 0:
                    v = -A[i][c]/float(pivot)
                    L[i][c] = -v 
                    for j in range(c, n):
                        A[i][j] = A[i][j] + v * float(A[d][j])
                        
            d = d + 1 

        return Matrix(L), Matrix(A), swap_lines

    # cholesky

    def cholesky(self):
        A = np.array(self.data)
        m = self.data.shape[0]
        G = np.zeros([m, m])

        #todo: verificar se eh positiva definida e simétrica

        for k in range(0, m):
            s = 0
            for i in range(0, k):
                s = s + (G[k, i] ** 2)
            
            s = A[k, k] - s

            if s <= 0:
                print("A matriz não eh positiva definida")
                break

            G[k, k] = math.sqrt(s)

            for j in range(k + 1, m):
                s = 0
                for i in range(0, k):
                    s = s + G[j, i] * G[k, i]
                
                G[j, k] = (A[j, k] - s) / G[k, k]

        return Matrix(G), Matrix(G.T)