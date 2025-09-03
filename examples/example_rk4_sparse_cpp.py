from rk4_sparse._rk4_sparse_cpp import rk4_sparse_eigen
import numpy as np
from scipy.sparse import csr_matrix



H0 = csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
    
# 双極子演算子
mux = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
muy = csr_matrix([[0, 0], [0, 0]], dtype=np.complex128)

# 初期状態（基底状態）
psi0 = np.array([1, 0], dtype=np.complex128)

# 共鳴パルス
steps = 100
t = np.linspace(0, 10, steps)
Ex = 0.1 * np.sin(t)  # 共鳴周波数
Ey = np.zeros_like(Ex)
dt = t[1] - t[0]

print(f"Time steps: {steps}")
print(f"Time step size: {dt:.3f}")
print(f"Total time: {t[-1]:.3f}")


# C++実装
print("Running C++ implementation...")
result_cpp = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt*2, True, 1, False)

# 結果の表示
pprint(f"Final state (C++):    [{result_cpp[-1, 0]:.3f}, {result_cpp[-1, 1]:.3f}]")

# 占有数の計算
pop_ground_cpp = np.abs(result_cpp[:, 0])**2
pop_excited_cpp = np.abs(result_cpp[:, 1])**2

print(f"\nFinal populations:")
print(f"C++     - Ground: {pop_ground_cpp[-1]:.6f}, Excited: {pop_excited_cpp[-1]:.6f}")
