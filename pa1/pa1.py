# imports
import numpy as np

# ----- exercise 2.c -----

import numpy as np

# original matrix
B = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]], dtype=float)

B_copy = B.copy()

# (i) double column 1
B1 = np.eye(4)
B1[0, 0] = 2
B_copy = B_copy @ B1

# (ii) halve row 3
B2 = np.eye(4)
B2[2, 2] = 0.5
B_copy = B2 @ B_copy

# (iii) add row 1 to row 4
B3 = np.eye(4)
B3[3, 0] = 1
B_copy = B3 @ B_copy

# (iv) interchange columns 2 and 3
B4 = np.eye(4)
B4[:, [1, 2]] = B4[:, [2, 1]]
B_copy = B_copy @ B4

# (v) subtract row 2 from each of the other rows
B5 = np.eye(4)
B5[0, 1] = -1
B5[2, 1] = -1
B5[3, 1] = -1
B_copy = B5 @ B_copy

# (vi) replace column 4 by column 1
B6 = np.eye(4)
B6[:, 3] = B6[:, 0]
B_copy = B_copy @ B6

# (vii) delete column 2
B7 = np.delete(np.eye(4), 1, axis=1)
D = B_copy @ B7

print("Final Matrix D:\n", D)

# row operations

## (ii) halve row 3
R_2 = np.eye(4)
R_2[2, 2] = 0.5

## (iii) add row 1 to row 4
R_3 = np.eye(4)
R_3[3, 0] = 1

## (v) subtract row 2 from each of the other rows
R_5 = np.eye(4)
R_5[0, 1] -= 1
R_5[2, 1] -= 1
R_5[3, 1] -= 1

## combine all row operations
A = R_5 @ R_3 @ R_2

# column operations

## (i) double column 1
C_1 = np.eye(4)
C_1[0, 0] = 2

## (iv) interchange columns 2 and 3
C_4 = np.eye(4)
C_4[:, [1, 2]] = C_4[:, [2, 1]]

## (vi) replace column 4 by column 1
C_6 = np.eye(4)
C_6[:, 3] = C_6[:, 0]

## (vii) delete column 2
C_7 = np.delete(np.eye(4), 1, axis=1)

## combine all column operations
C = C_1 @ C_4 @ C_6 @ C_7

# resulting matrix D
D = A @ B @ C

print("A =\n", A)
print("C =\n", C)
print("Final Matrix D = A @ B @ C =\n", D)

# ----- exercise 4.c -----
import numpy as np
import matplotlib.pyplot as plt

# setup mesh
x = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, x)

# compute norms
norm_inf = np.maximum(np.abs(X), np.abs(Y))
norm2 = np.sqrt(X**2 + Y**2)

# plot contours
plt.figure(figsize=(6,6))
plt.contour(X, Y, norm_inf, levels=[1], colors='red', linewidths=2, linestyles='--', label=r'$\|x\|_\infty=1$')
plt.contour(X, Y, norm2, levels=[1], colors='blue', linewidths=2, label=r'$\|x\|_2=1$')
# plt.contourf(X, Y, norm2, levels=[0, 1], colors=['#a8dadc'], alpha=0.5) # fill inside unit 2-norm circle
# plt.contourf(X, Y, norm2 > 1, levels=[0.5, 1.5], colors=['#e63946'], alpha=0.3) # fill outside unit 2-norm circle

# pretty plot
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r"Contours of $\|x\|_\infty=1$ and $\|x\|_2=1$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(True)
plt.show()

# ----- exercise 5.b -----

# imports
import numpy as np

# variable initialization
vector_a = np.array([[1],
                    [3],
                    [5]])

vector_b = np.array([[2],
                    [4],
                    [6]])

matrix_a = np.array([[1, 2, 3],
                    [4, 5, 6]])

matrix_b = np.array([[7, 8],
                    [9, 10],
                    [11, 12]])

matrix_c = np.array([[1, 0 , 0],
                    [0, 0, 1]])

beta_a, beta_b = 4, 5

## verify part (i), vector operations
vector_add = vector_a + vector_b
vector_scalar = beta_a * vector_a
dot_prod = np.dot(vector_a.T, vector_b)
linear_combo = (vector_scalar) + (beta_b * vector_b)
vector_operations_dict = {
    "vector_add": vector_add,
    "vector_scalar": vector_scalar,
    "dot_prod": dot_prod,
    "linear_combo": linear_combo
}

for k, v in vector_operations_dict.items():
    print(f"{k}:\n{v}\n")

## verify part (ii), matrix operations
matrix_scalar = beta_a * matrix_a
# matrix_add_b = matrix_a + matrix_b # incompatible matrix
matrix_add_ac = matrix_a + matrix_c
matrix_operations_dict = {
    "matrix_scalar": matrix_scalar,
    "matrix_add_ac": matrix_add_ac
}

for k, v in matrix_operations_dict.items():
    print(f"{k}:\n{v}\n")

## verify part (iii), transpose operations
transpose_prod = np.transpose(matrix_a @ matrix_b)
transpose_prod_new = transpose_prod
transpose_transpose = np.transpose(np.transpose(matrix_a))
transpose_sum = np.transpose(matrix_a + matrix_c)
transpose_dict = {
    "transpose_prod": transpose_prod,
    "transpose_prod_new": transpose_prod_new,
    "transpose_transpose": transpose_transpose,
    "transpose_sum": transpose_sum
}
for k, v in transpose_dict.items():
    print(f"{k}:\n{v}\n")

## verify part (iv), inner and outer products
vec_a_flat = vector_a.flatten()
vec_b_flat = vector_b.flatten()
inner_ab = np.dot(vec_a_flat, vec_b_flat)
inner_ba = np.dot(vec_b_flat, vec_a_flat)
inner_aa = np.dot(vec_a_flat, vec_a_flat)
inner_bb = np.dot(vec_b_flat, vec_b_flat)
scalar_out = beta_a * inner_ab
scalar_in = np.dot(beta_a * vec_a_flat, vec_b_flat)
outer_prod = np.outer(vec_b_flat, vec_a_flat)

inner_prod_dict = {
    "inner_ab": inner_ab,
    "inner_ba": inner_ba,
    "inner_aa": inner_aa,
    "inner_bb": inner_bb,
    "scalar_out": scalar_out,
    "scalar_in": scalar_in,
    "outer_prod": outer_prod
}

for k, v in inner_prod_dict.items():
    print(f"{k}:\n{v}\n")

# verify part (v), determinants
det_ab = np.linalg.det(matrix_a @ matrix_b)
det_bc = np.linalg.det(matrix_b @ matrix_c)
det_dict = {
    "det_ab": det_ab,
    "det_bc": det_bc
}

for k, v in det_dict.items():
    print(f"{k}:\n{v}\n")