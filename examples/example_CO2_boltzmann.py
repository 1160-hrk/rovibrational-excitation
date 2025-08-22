#!/usr/bin/env python
"""
回転振動励起シミュレーション例
=============================

線形分子の回転振動励起における時間発展シミュレーション。
振動-回転結合効果や異なる励起条件での応答を観察。

実行方法:
    python examples/example_rovibrational_excitation.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple, Any, cast
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.analysis.absorbance import compute_absorbance_spectrum

# %% パラメータ設定
# システムパラメータ
SPARSE = True
# SPARSE = False
DENSE = False
V_MAX = 10  # 最大振動量子数
J_MAX = 16  # 最大回転量子数
USE_M = True  # 磁気量子数を使用

# 分子パラメータ
OMEGA_01 = 2349.1  # 振動周波数 [cm^-1]
DOMEGA = 25  # 非調和性補正 [cm^-1]
B_CONSTANT = 0.39  # 回転定数 [cm^-1]
ALPHA_CONSTANT = 0.00307  # 振動-回転相互作用定数 [cm^-1]
ALPHA_CONSTANT = 0.0  # 振動-回転相互作用定数 [cm^-1]
UNIT_FREQUENCY = "cm^-1"
MU0 = 1e-30  # 双極子行列要素の大きさ [C·m]
UNIT_DIPOLE = "C*m"

# 温度設定（ボルツマン分布）
TEMPERATURE_K = 10.0
# ボルツマン重みの閾値（これ以上の重みを持つ準位のみ計算）
BOLTZMANN_WEIGHT_THRESHOLD = 1e-4

# レーザーパルス設定
PULSE_DURATION = 100.0  # パルス幅 [fs]
DETUNING = 0.0  # デチューニング
POLARIZATION = np.array([1, 0])  # x方向偏光
AXES = "xy"  # x, y方向の双極子を考慮

# フルエンスから電場振幅を算出（g(t)=exp(-(t-tc)^2/(2σ^2)) を使用）
# F = ∫ I(t) dt,  I(t) ≈ (ε0 c/2) |E0|^2 g(t)^2,  ∫ g(t)^2 dt = σ√π
FLUENCE_MJ_CM2 = 100.0  # [mJ/cm^2]
F_SI = FLUENCE_MJ_CM2 * 10.0  # [J/m^2]
sigma_s = converter.convert_time(PULSE_DURATION, "fs", "s")
eps0 = 8.8541878128e-12
c0 = 299792458.0
pol_norm2 = float(np.sum(POLARIZATION.astype(float) ** 2))
if pol_norm2 <= 0.0:
    pol_norm2 = 1.0
EFIELD_AMPLITUDE = np.sqrt(2.0 * F_SI / (eps0 * c0 * pol_norm2 * sigma_s * np.sqrt(np.pi)))
print(
    f"EFIELD_AMPLITUDE (from fluence): {EFIELD_AMPLITUDE:.3e} V/m "
    f"(F={FLUENCE_MJ_CM2} mJ/cm^2, sigma={PULSE_DURATION} fs)"
)

# 時間グリッド設定
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = PULSE_DURATION*10  # 終了時間 [fs]
DT_EFIELD = 0.1  # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 1  # サンプリングストライド


# plot setting
def plot_states():
    states = []
    
    for V in range(V_MAX+1):
        for J in range(J_MAX+1):
            for M in range(-J, J+1):
                if AXES == "xy":
                    if (V + J) % 2 == 0 and (J + M)%2 == 0:
                        states.append((V, J, M))
                elif AXES == "zx":
                    if (V + J) % 2 == 0 and M == 0:
                        states.append((V, J, M))
                else:
                    raise ValueError(f"Invalid axes: {AXES}")
    return states


# 並列設定（インタラクティブに調整可能）
USE_PARALLEL = True
# USE_PARALLEL = False
# PARALLEL_BACKEND = "thread"  # "thread" or "process"
PARALLEL_BACKEND = "process"  # "thread" or "process"
CHUNK_SIZE = 8
MAX_WORKERS = min(os.cpu_count() or 1, 14)
PROGRESS_EVERY = 1  # 進捗ログ出力のバッチ間隔


# 吸収スペクトル設定
# DETECTION_POL = np.array([1.0, 0.0])  # 検出偏光ベクトル (μ · e*)
DETECTION_POL = np.array([np.sqrt(1/3), np.sqrt(2/3)])  # マジックアングル 54.7°
WN_MIN, WN_MAX, WN_STEP = 2000.0, 2400.0, 0.01  # [cm^-1]
T2_PS = 667.0  # 緩和 [ps]
PATH_LENGTH_M = 1.0e-3  # [m]
PRESSURE_PA = 6.0e-2  # [Pa]


# 並列化用ユーティリティ
def chunked_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def compute_chunk_for_indices(
    indices_batch: list[int],
    boltz_weights: np.ndarray,
    basis_size: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    diag_chunk: np.ndarray | None = None
    rho_chunk = np.zeros((basis_size, basis_size), dtype=np.complex128)

    for idx_init in indices_batch:
        weight = float(boltz_weights[idx_init])
        if weight <= 0.0:
            continue

        psi0_i = np.zeros(basis_size, dtype=np.complex128)
        psi0_i[idx_init] = 1.0

        t_i_and_psi = schrodinger_propagation(
            hamiltonian=H0,
            Efield=Efield,
            dipole_matrix=dipole_matrix,
            psi0=psi0_i,
            axes=AXES,
            return_traj=True,
            return_time_psi=True,
            sample_stride=SAMPLE_STRIDE,
            sparse=SPARSE,
            algorithm="rk4",
        )
        t_i, psi_t_i = cast(Tuple[np.ndarray, np.ndarray], t_i_and_psi)

        prob_t_i = np.abs(psi_t_i) ** 2
        if diag_chunk is None:
            diag_chunk = np.zeros_like(prob_t_i)
        diag_chunk += weight * prob_t_i

        psi_final = psi_t_i[-1, :]
        rho_chunk += weight * np.outer(psi_final, np.conj(psi_final))

    return diag_chunk, rho_chunk


def run_boltzmann_ensemble_parallel(
    nonzero_indices: list[int],
    boltz_weights: np.ndarray,
    basis_size: int,
    *,
    chunk_size: int,
    max_workers: int,
    backend: str = "thread",
    progress_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    batches = chunked_indices(nonzero_indices, chunk_size)

    diag_sum: np.ndarray | None = None
    rho_sum = np.zeros((basis_size, basis_size), dtype=np.complex128)

    if USE_PARALLEL and len(batches) > 1:
        Executor = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    compute_chunk_for_indices,
                    batch,
                    boltz_weights,
                    basis_size,
                )
                for batch in batches
            ]
            completed = 0
            for fut in as_completed(futures):
                diag_chunk, rho_chunk = fut.result()
                if diag_chunk is not None:
                    if diag_sum is None:
                        diag_sum = np.zeros_like(diag_chunk)
                    diag_sum += diag_chunk
                rho_sum += rho_chunk
                completed += 1
                if progress_every > 0 and completed % progress_every == 0:
                    print(f"  Completed batches: {completed}/{len(batches)}")
    else:
        # フォールバック（逐次）
        for k, batch in enumerate(batches, start=1):
            diag_chunk, rho_chunk = compute_chunk_for_indices(batch, boltz_weights, basis_size)
            if diag_chunk is not None:
                if diag_sum is None:
                    diag_sum = np.zeros_like(diag_chunk)
                diag_sum += diag_chunk
            rho_sum += rho_chunk
            if progress_every > 0 and k % progress_every == 0:
                print(f"  Completed batches: {k}/{len(batches)}")

    assert diag_sum is not None, "No ensemble populations accumulated."
    return diag_sum, rho_sum

# %% 基底・ハミルトニアン・双極子行列の生成
print("=== 回転振動励起シミュレーション ===")
print(f"基底サイズ: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")

basis = LinMolBasis(
    V_max=V_MAX, J_max=J_MAX, use_M=USE_M,
    omega=OMEGA_01,
    delta_omega=DOMEGA,
    B=B_CONSTANT,
    alpha=ALPHA_CONSTANT,
    input_units=UNIT_FREQUENCY,
    output_units="rad/fs"
)
H0 = basis.generate_H0()

print(f"基底次元: {basis.size()}")
print(f"エネルギー準位数: {len(H0.get_eigenvalues())}")

# エネルギー準位の表示
eigenvalues = H0.get_eigenvalues()
print(f"最低エネルギー: {eigenvalues[0]:.6f} rad/fs")
print(f"最高エネルギー: {eigenvalues[-1]:.6f} rad/fs")

dipole_matrix = LinMolDipoleMatrix(
    basis=basis,
    mu0=MU0,
    potential_type="harmonic",
    backend="numpy",
    dense=DENSE,
    units="C*m",
    units_input=UNIT_DIPOLE,
)

# %% 初期状態の設定
state = StateVector(basis)
state.set_state((0, 0, 0), 1.0)  # 基底状態 |v=0, J=0, M=0⟩
psi0 = state.data

print(f"初期状態: |v=0, J=0, M=0⟩ (インデックス: {basis.get_index((0, 0, 0))})")

# %% 時間グリッド・電場生成
# 共鳴ケース


time4Efield = np.arange(TIME_START, TIME_END + 2 * DT_EFIELD, DT_EFIELD)
tc = (time4Efield[-1] + time4Efield[0]) / 2

carrier_freq = converter.convert_frequency(
    H0.eigenvalues[basis.get_index((4, 1, 1))] - H0.eigenvalues[basis.get_index((3, 0, 0))],
    "rad/fs", UNIT_FREQUENCY
    )
Efield = ElectricField(tlist=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=PULSE_DURATION,
    t_center=tc,
    carrier_freq=float(carrier_freq),
    carrier_freq_units=UNIT_FREQUENCY,
    amplitude=EFIELD_AMPLITUDE,
    polarization=POLARIZATION,  # x方向偏光
    const_polarisation=False,
)

# %% ボルツマン分布アンサンブルの時間発展
print(f"=== Rovibrational Excitation with Boltzmann ensemble (T={TEMPERATURE_K:.1f} K) ===")

# ボルツマン重みの計算（E = ħ ω, ω [rad/fs] → [rad/s] に変換してエネルギー[J]へ）
eigenvalues = H0.get_eigenvalues()
omega_min = np.min(eigenvalues)
omega_shift = eigenvalues - omega_min  # 最低エネルギーを0にシフト
HBAR_SI = CONSTANTS.HBAR  # [J*s]
KB_SI = 1.380649e-23  # [J/K]

# 温度0 Kのときは基底状態のみ（数値的に安定化）
if TEMPERATURE_K <= 0.0:
    boltz_weights = np.zeros_like(omega_shift)
    boltz_weights[np.argmin(eigenvalues)] = 1.0
    Z_value = 1.0
else:
    energies_J = HBAR_SI * (omega_shift * 1e15)  # [J]
    boltz_raw = np.exp(-energies_J / (KB_SI * TEMPERATURE_K))
    Z = float(np.sum(boltz_raw))
    boltz_weights = boltz_raw / max(Z, 1e-300)
    Z_value = Z

print(f"Partition function (Z): {Z_value:.6e}")

# 対称分子 CO2 の制約: J は偶数のみ許容 → 奇数Jの重みを0にし再正規化
even_J_mask = np.array([(1.0 if (state[1] % 2 == 0) else 0.0) for state in basis.basis])
boltz_weights = boltz_weights * even_J_mask
sum_allowed = float(np.sum(boltz_weights))
if sum_allowed > 0.0:
    boltz_weights = boltz_weights / sum_allowed
else:
    # フォールバック: J=0 が存在すればそこに全集中
    try:
        j0_idx = basis.get_index((0, 0, 0))
    except Exception:
        j0_idx = int(np.argmin(eigenvalues))
    boltz_weights[:] = 0.0
    boltz_weights[j0_idx] = 1.0

# 閾値以上の初期状態インデックス（偶数Jのみ）
nonzero_indices = []
for i in range(basis.size()):
    v_i, J_i, M_i = basis.basis[i]
    if (J_i % 2 == 0) and (float(boltz_weights[i]) >= float(BOLTZMANN_WEIGHT_THRESHOLD)):
        nonzero_indices.append(int(i))

if len(nonzero_indices) == 0:
    # フォールバック: 最も重みの大きい準位を1つだけ採用
    fallback_idx = int(np.argmax(boltz_weights))
    nonzero_indices = [fallback_idx]
    kept_weight = float(boltz_weights[fallback_idx])
    print(
        f"No states above threshold (threshold={BOLTZMANN_WEIGHT_THRESHOLD:.2e}). "
        f"Fallback to the max-weight state idx={fallback_idx} (kept_weight={kept_weight:.3e})."
    )
else:
    kept_weight = float(np.sum(boltz_weights[nonzero_indices]))
    print(
        f"Selected states: {len(nonzero_indices)} / {basis.size()} "
        f"(threshold={BOLTZMANN_WEIGHT_THRESHOLD:.2e}, kept_weight={kept_weight:.3e})"
    )

# ウォームアップ（時間配列 shape 確定）
warm_idx = nonzero_indices[0]
psi0_warm = np.zeros(basis.size(), dtype=np.complex128)
psi0_warm[warm_idx] = 1.0
t_and_psi_warm = schrodinger_propagation(
    hamiltonian=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,
    psi0=psi0_warm,
    axes=AXES,
    return_traj=True,
    return_time_psi=True,
    sample_stride=SAMPLE_STRIDE,
    sparse=SPARSE,
    algorithm="rk4",
)
time4psi, psi_t_warm = cast(Tuple[np.ndarray, np.ndarray], t_and_psi_warm)

# 並列実行（または逐次フォールバック）
start = time.perf_counter()
diag_rho_t, rho_final = run_boltzmann_ensemble_parallel(
    nonzero_indices=nonzero_indices,
    boltz_weights=boltz_weights,
    basis_size=basis.size(),
    chunk_size=CHUNK_SIZE,
    max_workers=MAX_WORKERS,
    backend=PARALLEL_BACKEND,
    progress_every=PROGRESS_EVERY,
)
end = time.perf_counter()
print(f"Ensemble propagation finished. Elapsed: {end - start:.3f} s")


# %% 結果保存（最終状態の密度行列）
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
rho_path = os.path.join(results_dir, f"rho_final_{timestamp}.npy")
np.save(rho_path, rho_final)
print(f"Saved final density matrix to: {rho_path}")


# %% 結果プロット
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 電場の時間発展
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], "r-", linewidth=1.5, label=r"$E_x(t)$")
axes[0].plot(time4Efield, Efield_data[:, 1], "b-", linewidth=1.5, label=r"$E_y(t)$")
axes[0].set_ylabel("Electric Field [a.u.]")
axes[0].set_title(f"Rovibrational Excitation, δ={DETUNING:.3f}, E={EFIELD_AMPLITUDE:.3e} V/m")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 各状態の占有確率（アンサンブル）
assert diag_rho_t is not None, "No ensemble populations accumulated. Check temperature and weights."
assert time4psi is not None, "No time grid from propagation."
prob_t = diag_rho_t
total_prob = np.sum(prob_t, axis=1)

# 主要な状態のみをプロット
main_states = plot_states()

for i, (v, J, M) in enumerate(main_states):
    if (v, J, M) in basis.index_map:
        idx = basis.get_index((v, J, M))
        axes[1].plot(time4psi, prob_t[:, idx], linewidth=2, 
                    label=rf"$|v={v}, J={J}, M={M}\rangle$")

axes[1].plot(time4psi, total_prob, "k--", alpha=0.7, linewidth=1, label="Total")
axes[1].set_ylabel("Population")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# 振動準位別の占有確率
vib_probs = {}
for i, (v, J, M) in enumerate(basis.basis):
    if v not in vib_probs:
        vib_probs[v] = np.zeros_like(time4psi)
    vib_probs[v] += prob_t[:, i]

for v in sorted(vib_probs.keys()):
    axes[2].plot(time4psi, vib_probs[v], linewidth=2, label=rf"$v={v}$")

axes[2].set_ylabel("Vibrational Population")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.05, 1.05)

# 回転準位別の占有確率
rot_probs = {}
for i, (v, J, M) in enumerate(basis.basis):
    if J not in rot_probs:
        rot_probs[J] = np.zeros_like(time4psi)
    rot_probs[J] += prob_t[:, i]

for J in sorted(rot_probs.keys()):
    axes[3].plot(time4psi, rot_probs[J], linewidth=2, label=rf"$J={J}$")

axes[3].set_xlabel("Time [fs]")
axes[3].set_ylabel("Rotational Population")
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()


# %% 吸収スペクトルの計算 (2000–2400 cm^-1, T2=600 ps)
# Hamiltonian を [J] に変換
H0_J = H0.to_energy_units()
H0_matrix_J = H0_J.get_matrix() if hasattr(H0_J, "get_matrix") else H0_J.matrix

# 双極子（SI, C·m）: x, y 成分を取得して検出偏光で合成（密行列で取得）
try:
    mu_x = dipole_matrix.get_mu_x_SI(dense=True)
    mu_y = dipole_matrix.get_mu_y_SI(dense=True)
except AttributeError:
    # 後方互換
    mu_x = dipole_matrix.mu("x", dense=True)
    mu_y = dipole_matrix.mu("y", dense=True)

mu_int = POLARIZATION[0] * mu_x + POLARIZATION[1] * mu_y
mu_det = DETECTION_POL[0] * mu_x + DETECTION_POL[1] * mu_y  # 検出（複素共役は外で不要）

# スペクトル軸
nu_tilde = np.arange(WN_MIN, WN_MAX + 0.5 * WN_STEP, WN_STEP)

# 数密度 n = p/(k_B T)
KB_SI = 1.380649e-23
number_density = PRESSURE_PA / (KB_SI * TEMPERATURE_K)

print("Computing absorbance spectrum...")
# 初期密度行列（ボルツマン分布の対角行列）
rho_init = np.diag(boltz_weights.astype(np.complex128))
rho_for_spec = rho_final - rho_init
rho_for_spec = rho_init
res_spec = compute_absorbance_spectrum(
    rho=rho_for_spec,
    mu_int=mu_int,
    mu_det=mu_det,
    H0=H0_matrix_J,
    nu_tilde_cm=nu_tilde,
    T2=T2_PS * 1e-12,  # ps -> s
    number_density=number_density,
    path_length=PATH_LENGTH_M,
    local_field=True,
    return_resp=False,
    use_sparse_transitions=True,
    mu_nz_threshold=0.0,
    omega_chunk_size=5000,
)

# プロット
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
ax2.plot(res_spec["nu_tilde_cm"], res_spec["A_mOD"], "k-")
ax2.set_xlabel("Wavenumber [cm$^{-1}$]")
ax2.set_ylabel("Absorbance [mOD]")
ax2.set_title("Absorbance spectrum (T2 = %.1f ps)" % T2_PS)
ax2.grid(True, alpha=0.3)
# ax2.set_xlim(2300, 2400)
plt.tight_layout()

# 保存
absorb_path = os.path.join(results_dir, f"absorbance_{timestamp}.npz")
np.savez_compressed(
    absorb_path,
    nu_tilde_cm=res_spec["nu_tilde_cm"],
    omega=res_spec["omega"],
    A_mOD=res_spec["A_mOD"],
)
print(f"Saved absorbance spectrum to: {absorb_path}")
# %%
