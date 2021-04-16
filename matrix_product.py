import numpy as np
from lib.matrix import Matrix
from lib.linear_system import gauss_elimination, gauss_elimination_reduction
from lib.decompositions import QR_method_gram_sh, eign_qr_hh

np.set_printoptions(precision=5)

a = np.array([[4, 3], [6, 3]]) # 2x2
b = np.array([[3, 3, 1], [4, 5, 9]]) # 2x3
c = np.array([[2, 1, 1, 3], [3, 7, 9, 1], [1, 4, -11, 10]]) # 3x4
d = np.array([[2, 1, 0, 3, 4], [1, 1, -1, 1, 1], [1, -1, -1, 2, -3], [-3, 2, 3, -1, 4]]) # 4x5
e = np.array([[3, 4, 1], [2, 2, 1], [3, 4, 5]]) # 3x3
f = np.array([[4, 2, -4], [2, 10, 4], [-4, 4, 9]]) # 3x3 - positiva definida e simétrica
g = np.array([[3.0, 2.0, 4.0], [1.0, 1.0, 2.0], [4.0, 3.0, -2.0]]) # 3x3 - teste fatoração LU

ma = Matrix(a)
mb = Matrix(b)
mc = Matrix(c)
md = Matrix(d)
me = Matrix(e)
mf = Matrix(f)
mg = Matrix(g)

# test scalar product

#scalar_prod_a = ma.scalar_product(2)
#scalar_prod_b = mb.scalar_product(3)

# print(scalar_prod_a.data)
# print(scalar_prod_b.data)

# test matrix product

# matrix_product = ma.matrix_product(mc)

# test echelon 


# matrix_echelon = me.partial_pivot_echelon()
# print(matrix_echelon.data)
# matrix_echelon_reduced = me.partial_pivot_echelon_reduced()
# print(matrix_echelon_reduced.data)

# print(md.data)

# output = gauss_elimination(md)
# output = gauss_elimination_reduction(md)

# print(output)

# output = Matrix(output.T)
# A = Matrix(md.data[:, 0:md.data.shape[1] - 1])
# res = A.matrix_product(output)

# print(res.data)
# print(md.data[:, -1])


# HOUSE HOLDER
T = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
mt = Matrix(T)

HH, Q = mt.house_holder()
print(HH)
print(Q)

# print(np.matmul(Q, HH))

D, eign = eign_qr_hh(mt)

print(D)