# imports
import numpy as np

# ----- exercise 2.c -----

# variable initalization
b = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]], dtype=float)

# verify part a
## double column 1
b_i = b.copy()
b_i[:, 0] *= 2

## halve row 3
b_ii = b_i.copy()
b_ii[2] /= 2

## add row 1 to row 4
b_iii = b_ii.copy()
b_iii[3] += b_iii[0]

## interchange columns 2 and 3
b_iv = b_iii.copy()
b_iv[:, [1, 2]] = b_iv[:, [2, 1]]

## subtract row 2 from each of the other rows
b_v = b_iv.copy()
b_v[[0, 2, 3]] -= b_v[1]

## replace column 4 by column 1
b_vi = b_v.copy()
b_vi[:, 3] = b_vi[:, 0]

## delete column 2 (so that the column dimension is reduced by 1)
b_vii = np.delete(b_vi, 1, axis=1)

## compute matrix D, product of 8 matricies b to b_vii
d = b @ b_i @ b_ii @ b_iii @ b_iv @ b_v @ b_vi @ b_vii

q2_dict = {
    "b": b,
    "b_i": b_i,
    "b_ii": b_ii,
    "b_iii": b_iii,
    "b_iv": b_iv,
    "b_v": b_v,
    "b_vi": b_vi,
    "b_vii": b_vii,
    "d": d
}
for k, v in q2_dict.items():
    print(f"{k}:\n{v}\n")

# verify part b
    # imports
    import numpy as np

    # variable initalization
    b = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]], dtype=float)

    # verify part a
    ## double column 1
    b_i = b.copy()
    b_i[:, 0] *= 2

    ## halve row 3
    b_ii = b_i.copy()
    b_ii[2] /= 2

    ## add row 1 to row 4
    b_iii = b_ii.copy()
    b_iii[3] += b_iii[0]

    ## interchange columns 2 and 3
    b_iv = b_iii.copy()
    b_iv[:, [1, 2]] = b_iv[:, [2, 1]]

    ## subtract row 2 from each of the other rows
    b_v = b_iv.copy()
    b_v[[0, 2, 3]] -= b_v[1]

    ## replace column 4 by column 1
    b_vi = b_v.copy()
    b_vi[:, 3] = b_vi[:, 0]

    ## delete column 2 (so that the column dimension is reduced by 1)
    b_vii = np.delete(b_vi, 1, axis=1)

    ## compute matrix D, product of 8 matricies b to b_vii
    d = b @ b_i @ b_ii @ b_iii @ b_iv @ b_v @ b_vi @ b_vii

    q2_dict = {
        "b": b,
        "b_i": b_i,
        "b_ii": b_ii,
        "b_iii": b_iii,
        "b_iv": b_iv,
        "b_v": b_v,
        "b_vi": b_vi,
        "b_vii": b_vii,
        "d": d
    }
    for k, v in q2_dict.items():
        print(f"{k}:\n{v}\n")

    # verify part (b)
    ## elementary row operations
    A = np.eye(4)
    A[2,2] = 0.5 # halve row 3
    A[3,0] = 1 # add row 1 to row 4
    A[[0,2,3],1] = -1 # subtract row 2 from each of the other rows

    ## elementary column operations
    C = np.eye(4)
    C[0,0] = 2 # double column 1
    C[:, [1,2]] = C[:, [2,1]] # interchange columns 2 and 3
    C[:,3] = C[:,0] # replace column 4 by column 1

    ## delete column 2 to compute final matrix D
    D_full = A @ b @ C # because it was lowercase earlier
    D = np.delete(D_full, 1, axis=1)
    print("Final matrix D:\n", D)

# ----- exercise 4.c -----
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, x)
norm_inf = np.maximum(np.abs(X), np.abs(Y))
norm2 = np.sqrt(X**2 + Y**2)

plt.figure(figsize=(6,6))
plt.contour(X, Y, norm_inf, levels=[1], colors='black', linewidths=2, label='||x||_∞=1')
plt.contour(X, Y, norm2, levels=[1], colors='blue', linewidths=2, label='||x||₂=1')
plt.contourf(X, Y, norm2, levels=[0,1], colors=['#a8dadc'], alpha=0.5)  # ||x||₂ < 1 region
plt.contourf(X, Y, norm2, levels=[1,2], colors=['#e63946'], alpha=0.3)  # ||x||₂ > 1 region

plt.gca().set_aspect('equal', adjustable='box')
plt.title(r"Contours of $\|x\|_\infty=1$ and $\|x\|_2$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(True)
plt.show()

# ----- exercise 5.b -----

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
dot_prod = vector_a * vector_b
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
inner_ab = np.inner(vector_a, vector_b)
inner_ba = np.inner(vector_b, vector_a)
inner_aa = np.inner(vector_a, vector_a)
inner_bb = np.inner(vector_b, vector_b)
scalar_out = beta_a * inner_ab
scalar_in = np.inner((beta_a * vector_a), vector_b)
outer_prod = vector_b @ np.transpose(vector_a)
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