import numpy as np

try:
    from numba import njit
except ImportError:  # numba 不在でも動くダミー

    def njit(*args, **kwargs):  # type: ignore
        def deco(f):
            return f

        return deco

@njit(cache=True, fastmath=True)
def to_csr(A, tol=0.0):
    m, n = A.shape
    # 1st pass: nnz カウント
    nnz = 0
    for i in range(m):
        for j in range(n):
            a = A[i, j]
            if a != 0.0 and (tol == 0.0 or (a.real*a.real + a.imag*a.imag) > tol*tol):
                nnz += 1

    data = np.empty(nnz, np.complex128)
    indices = np.empty(nnz, np.int64)
    indptr = np.empty(m+1, np.int64)

    # 2nd pass: 詰める
    k = 0
    indptr[0] = 0
    for i in range(m):
        for j in range(n):
            a = A[i, j]
            if a != 0.0 and (tol == 0.0 or (a.real*a.real + a.imag*a.imag) > tol*tol):
                data[k] = a
                indices[k] = j
                k += 1
        indptr[i+1] = k

    return data, indices, indptr



@njit(cache=True, fastmath=True)
def csr_matvec(data, indices, indptr, x, y):
    # y[:] = A @ x
    n = indptr.size - 1
    for i in range(n):
        s = 0.0 + 0.0j
        row_start = indptr[i]
        row_end   = indptr[i+1]
        for k in range(row_start, row_end):
            s += data[k] * x[indices[k]]
        y[i] = s

@njit(cache=True, fastmath=True)
def axpy_inplace(alpha, x, y):
    # y[:] += alpha * x
    n = y.size
    for i in range(n):
        y[i] += alpha * x[i]


@njit(cache=True, fastmath=True)
def H_apply( # y = (H0 - ex*mux - ey*muy) @ x
    H0_data, H0_idx, H0_ptr,
    mx_data, mx_idx, mx_ptr,
    my_data, my_idx, my_ptr,
    ex, ey,
    x, y, tmp0, tmpx, tmpy
):
    # tmp* は作業用。呼び出し側で再利用する
    csr_matvec(H0_data, H0_idx, H0_ptr, x, tmp0)
    csr_matvec(mx_data, mx_idx, mx_ptr, x, tmpx)
    csr_matvec(my_data, my_idx, my_ptr, x, tmpy)

    # y = tmp0 - ex*tmpx - ey*tmpy
    n = y.size
    for i in range(n):
        y[i] = tmp0[i] - ex*tmpx[i] - ey*tmpy[i]
