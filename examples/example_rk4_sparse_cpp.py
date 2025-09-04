# %%

from rk4_sparse import rk4_sparse_eigen_cached
import rk4_sparse._rk4_sparse_cpp as m; m.get_omp_max_threads()

import numpy as np
from scipy.sparse import csr_matrix

# より小さなテストケースで開始
dim = 2
num_steps = 10  # ステップ数を減らす
H0 = csr_matrix(np.diag(np.arange(dim)), dtype=np.complex128)
mux = csr_matrix(np.eye(dim, k=1) + np.eye(dim, k=-1), dtype=np.complex128)
muy = csr_matrix((dim, dim), dtype=np.complex128)

# 初期状態
psi0 = np.zeros(dim, dtype=np.complex128)
psi0[0] = 1.0

# 電場パラメータ（より小さな値）
dt_E = 0.1  # より大きな時間ステップ
E0 = 0.01   # より小さな電場振幅
omega_L = 0.1  # より小さな周波数
t = np.arange(0, dt_E * (num_steps+2), dt_E)
Ex = E0 * np.sin(omega_L * t)
Ey = np.zeros_like(Ex)

print(f"Time steps: {num_steps}")
print(f"Time step size: {dt_E:.3f}")
print(f"Total time: {t[-1]:.3f}")
print(f"Matrix shapes: H0={H0.shape}, mux={mux.shape}, muy={muy.shape}")
print(f"Field shapes: Ex={Ex.shape}, Ey={Ey.shape}")
print(f"Initial state shape: {psi0.shape}")

# %% C++実装
print("Running C++ implementation...")
result = rk4_sparse_eigen_cached(H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)