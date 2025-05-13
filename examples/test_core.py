# test_core.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core.basis import VJMBasis
from core.states import StateVector, DensityMatrix
from core.hamiltonian import generate_free_hamiltonian, generate_dipole_matrix
from core.electric_field import ElectricField
from core.propagator import schrodinger_propagation, liouville_propagation

print("=== 基底のテスト ===")
basis = VJMBasis(V_max=1, J_max=1)
print(f"基底サイズ: {basis.size()}")
print(f"例: index=0 -> {basis.get_state(0)}")

print("\n=== 状態ベクトルのテスト ===")
sv = StateVector(basis)
sv.set_state(0, 0, 0)
sv.normalize()
print(sv)

print("\n=== 密度行列のテスト ===")
dm = DensityMatrix(basis)
dm.set_pure_state(sv)
dm.normalize()
print(dm)

print("\n=== ハミルトニアン生成（物理モデル） ===")
H0 = generate_free_hamiltonian(
    basis,
    h=1.0,
    omega=0.2 * np.pi,
    delta_omega=0.01 * np.pi,
    B=0.02,
    alpha=0.001
)
print(f"H0.shape: {H0.shape}")

# 遷移双極子モーメント関数（簡易な選択則）
def dummy_tdm(q1, q2):
    dv = abs(q1[0] - q2[0])
    dj = abs(q1[1] - q2[1])
    dm = abs(q1[2] - q2[2])
    if dv == 1 and dj <= 1 and dm <= 1:
        return 1.0
    return 0.0

mu = generate_dipole_matrix(basis, dummy_tdm)
print(f"mu.shape: {mu.shape}, 非ゼロ要素数: {np.count_nonzero(mu)}")

print("\n=== 電場生成 ===")
tlist = np.linspace(-100, 100, 1000)
envelope = lambda t: np.exp(-(t/30)**2)
Efield = ElectricField(tlist, envelope_func=envelope, carrier_freq=0.1*np.pi)
Efield.plot()

print("\n=== シュレディンガー方程式の時間発展 ===")
H_list = [mu * E for E in Efield.E_real]
psi_t = schrodinger_propagation(H0, H_list, sv.data, tlist)
print(f"psi_t.shape = {psi_t.shape}")

print("\n=== リウヴィル方程式の時間発展 ===")
rho_t = liouville_propagation(H0, H_list, dm.data, tlist)
print(f"rho_t.shape = {rho_t.shape}")
