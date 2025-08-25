#!/usr/bin/env python
"""
ボルツマン分布アンサンブルの振動回転励起と吸収変化スペクトル計算
=================================================================

新しいAbsorbanceCalculatorクラスを使用して、
ボルツマン分布の初期状態から振動回転励起後の吸収変化を計算。

実行方法:
    python examples/example_CO2_boltzmann_with_calculator.py
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
from rovibrational_excitation.spectroscopy import (
    AbsorbanceCalculator,
    ExperimentalConditions,
    create_calculator_from_params
)

# %% パラメータ設定

# ===== 並列処理設定 =====
USE_PARALLEL = True
PARALLEL_BACKEND = "process"  # "thread" or "process"
CHUNK_SIZE = 12
MAX_WORKERS = min(os.cpu_count() or 1, 14)
PROGRESS_EVERY = 1  # 進捗ログ出力のバッチ間隔

# ===== システムパラメータ =====
SPARSE = True
DENSE = False
V_MAX = 11  # 最大振動量子数
J_MAX = 2  # 最大回転量子数
USE_M = True  # 磁気量子数を使用

# ===== 分子パラメータ (CO2) =====
OMEGA_01 = 2349.1  # 振動周波数 [cm^-1]
DOMEGA = 25  # 非調和性補正 [cm^-1]
B_CONSTANT = 0.39  # 回転定数 [cm^-1]
ALPHA_CONSTANT = 0.0  # 振動-回転相互作用定数 [cm^-1] (簡単のため0)
UNIT_FREQUENCY = "cm^-1"
MU0 = 1e-30  # 双極子行列要素の大きさ [C·m]
UNIT_DIPOLE = "C*m"

# ===== 温度設定（ボルツマン分布） =====
TEMPERATURE_K = 10.0
BOLTZMANN_WEIGHT_THRESHOLD = 1e-4  # この重み以上の準位のみ計算

# ===== レーザーパルス設定 =====
PULSE_DURATION_FWHM = 100.0  # パルス幅 [fs]
PULSE_DURATION = PULSE_DURATION_FWHM / (2.0 * np.sqrt(np.log(2.0)))  # パルス幅 [fs]
DETUNING = 0.0  # デチューニング
GDD = -2000.0  # [fs^2]
POLARIZATION = np.array([1, 0])  # x方向偏光
AXES = "xy"  # x, y方向の双極子を考慮

# キャリア周波数を設定するための状態ペア（下位, 上位）
CARRIER_STATE_LOWER = (4, 0, 0)
CARRIER_STATE_UPPER = (5, 1, 1)

# ===== フルエンス設定 =====
FLUENCE_MJ_CM2 = 100.0  # [mJ/cm^2]
F_SI = FLUENCE_MJ_CM2 * 10.0  # [J/m^2]

# ===== 時間グリッド設定 =====
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = PULSE_DURATION * 10  # 終了時間 [fs]
DT_EFIELD = 0.05  # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 1  # サンプリングストライド

# ===== 吸収スペクトル設定 =====
# 実験条件
TEMPERATURE_SPEC = TEMPERATURE_K  # スペクトル計算時の温度 [K]
PRESSURE_PA = 6.0e-1  # 圧力 [Pa]
PATH_LENGTH_M = 1.0e-3  # 光路長 [m]
T2_PS = 667.0  # コヒーレンス緩和時間 [ps]

# 偏光設定
# INTERACTION_POL = np.array([1.0, 0.0])  # 相互作用光の偏光
INTERACTION_POL = np.array([np.sqrt(1/3), np.sqrt(2/3)])  # 相互作用光の偏光
DETECTION_POL = np.array([np.sqrt(1/3), np.sqrt(2/3)])  # マジックアングル検出 54.7°

# スペクトル範囲
WN_MIN, WN_MAX, WN_STEP = 2100.0, 2400.0, 0.01  # [cm^-1]

# ===== プロット設定 =====
def get_plot_states():
    """Select main states for plotting"""
    states = []
    for V in range(min(V_MAX+1, 3)):  # First 3 vibrational levels
        for J in range(min(J_MAX+1, 5)):  # First 5 rotational levels
            for M in range(-J, J+1):
                if AXES == "xy":
                    if (V + J) % 2 == 0 and (J + M) % 2 == 0:
                        states.append((V, J, M))
                elif AXES == "zx":
                    if (V + J) % 2 == 0 and M == 0:
                        states.append((V, J, M))
                else:
                    raise ValueError(f"Invalid axes: {AXES}")
    return states[:10]  # Maximum 10 states


# %% ユーティリティ関数

def _state_for_basis(state_tuple: tuple[int, int, int]) -> tuple:
    """Adjust state representation according to USE_M setting"""
    return state_tuple if USE_M else (state_tuple[0], state_tuple[1])


def chunked_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    """Split index list into chunks"""
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def compute_chunk_for_indices(
    indices_batch: list[int],
    boltz_weights: np.ndarray,
    basis_size: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Compute time evolution for a batch of indices (for parallel processing)"""
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
    """Parallel time evolution calculation for Boltzmann ensemble"""
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
        # 逐次実行
        for k, batch in enumerate(batches, start=1):
            diag_chunk, rho_chunk = compute_chunk_for_indices(batch, boltz_weights, basis_size)
            if diag_chunk is not None:
                if diag_sum is None:
                    diag_sum = np.zeros_like(diag_chunk)
                diag_sum += diag_chunk
            rho_sum += rho_chunk
            if progress_every > 0 and k % progress_every == 0:
                print(f"  Completed batches: {k}/{len(batches)}")

    assert diag_sum is not None, "No ensemble populations were calculated"
    return diag_sum, rho_sum


# %% 基底・ハミルトニアン・双極子行列の生成
print("=" * 70)
print("Rovibrational Excitation Simulation with Boltzmann Ensemble")
print("=" * 70)
print(f"Basis size: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")

# 基底の生成
basis = LinMolBasis(
    V_max=V_MAX, 
    J_max=J_MAX, 
    use_M=USE_M,
    omega=OMEGA_01,
    delta_omega=DOMEGA,
    B=B_CONSTANT,
    alpha=ALPHA_CONSTANT,
    input_units=UNIT_FREQUENCY,
    output_units="J"  # AbsorbanceCalculatorはJ単位を期待
)

# ハミルトニアンの生成
H0 = basis.generate_H0()
print(f"Basis dimension: {basis.size()}")

# エネルギー準位の表示
eigenvalues = H0.get_eigenvalues(units="J")
print(f"Lowest energy: {eigenvalues[0]:.6e} J")
print(f"Highest energy: {eigenvalues[-1]:.6e} J")

# 双極子行列の生成
dipole_matrix = LinMolDipoleMatrix(
    basis=basis,
    mu0=MU0,
    potential_type="harmonic",
    backend="numpy",
    dense=DENSE,
    units="C*m",
    units_input=UNIT_DIPOLE,
)

# %% 電場の生成

# 電場振幅の計算（フルエンスから）
sigma_s = converter.convert_time(PULSE_DURATION, "fs", "s")
eps0 = 8.8541878128e-12
c0 = 299792458.0
pol_norm2 = float(np.sum(POLARIZATION.astype(float) ** 2))
if pol_norm2 <= 0.0:
    pol_norm2 = 1.0
EFIELD_AMPLITUDE = np.sqrt(2.0 * F_SI / (eps0 * c0 * pol_norm2 * sigma_s * np.sqrt(np.pi)))
print(f"\nElectric field amplitude: {EFIELD_AMPLITUDE:.3e} V/m (Fluence={FLUENCE_MJ_CM2} mJ/cm^2)")

# 時間グリッド
time4Efield = np.arange(TIME_START, TIME_END + 2 * DT_EFIELD, DT_EFIELD)
tc = (time4Efield[-1] + time4Efield[0]) / 2

# キャリア周波数の計算
H0_rad_fs = basis.generate_H0(units="rad/fs")
carrier_freq = converter.convert_frequency(
    H0_rad_fs.eigenvalues[basis.get_index(_state_for_basis(CARRIER_STATE_UPPER))]
    - H0_rad_fs.eigenvalues[basis.get_index(_state_for_basis(CARRIER_STATE_LOWER))],
    "rad/fs", UNIT_FREQUENCY
)

# 電場の生成
Efield = ElectricField(tlist=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=PULSE_DURATION,
    t_center=tc,
    carrier_freq=float(carrier_freq),
    carrier_freq_units=UNIT_FREQUENCY,
    amplitude=EFIELD_AMPLITUDE,
    polarization=POLARIZATION,
    const_polarisation=False,
    gdd=GDD,
)

# %% ボルツマン分布の計算
print(f"\n=== Boltzmann Ensemble (T={TEMPERATURE_K:.1f} K) ===")

# ボルツマン重みの計算
omega_min = np.min(eigenvalues)
omega_shift = eigenvalues - omega_min  # 最低エネルギーを0にシフト
HBAR_SI = CONSTANTS.HBAR  # [J*s]
KB_SI = 1.380649e-23  # [J/K]

if TEMPERATURE_K <= 0.0:
    boltz_weights = np.zeros_like(omega_shift)
    boltz_weights[np.argmin(eigenvalues)] = 1.0
    Z_value = 1.0
else:
    energies_J = omega_shift  # すでにJ単位
    boltz_raw = np.exp(-energies_J / (KB_SI * TEMPERATURE_K))
    Z = float(np.sum(boltz_raw))
    boltz_weights = boltz_raw / max(Z, 1e-300)
    Z_value = Z

print(f"Partition function Z: {Z_value:.6e}")

# CO2の対称性制約: J偶数のみ許容
even_J_mask = np.array([(1.0 if (state[1] % 2 == 0) else 0.0) for state in basis.basis])
boltz_weights = boltz_weights * even_J_mask
sum_allowed = float(np.sum(boltz_weights))
if sum_allowed > 0.0:
    boltz_weights = boltz_weights / sum_allowed
else:
    # フォールバック
    try:
        j0_idx = basis.get_index((0, 0, 0))
    except Exception:
        j0_idx = int(np.argmin(eigenvalues))
    boltz_weights[:] = 0.0
    boltz_weights[j0_idx] = 1.0

# 閾値以上の初期状態インデックス
nonzero_indices = []
for i in range(basis.size()):
    v_i, J_i, M_i = basis.basis[i]
    if (J_i % 2 == 0) and (float(boltz_weights[i]) >= float(BOLTZMANN_WEIGHT_THRESHOLD)):
        nonzero_indices.append(int(i))

if len(nonzero_indices) == 0:
    fallback_idx = int(np.argmax(boltz_weights))
    nonzero_indices = [fallback_idx]
    kept_weight = float(boltz_weights[fallback_idx])
    print(f"No states above threshold. Fallback to max weight state (idx={fallback_idx})")
else:
    kept_weight = float(np.sum(boltz_weights[nonzero_indices]))
    print(f"Selected states: {len(nonzero_indices)} / {basis.size()} (kept weight={kept_weight:.3e})")

# %% 時間発展計算

# ウォームアップ（時間配列の形状確定）
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

# 並列実行
print("\nStarting time evolution calculation...")
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
print(f"Time evolution completed. Elapsed time: {end - start:.3f} s")

# %% AbsorbanceCalculatorを使用した吸収スペクトル計算
print("\n=== Absorbance Spectrum Calculation with AbsorbanceCalculator ===")

# 実験条件の設定
conditions = ExperimentalConditions(
    temperature=TEMPERATURE_SPEC,
    pressure=PRESSURE_PA,
    optical_length=PATH_LENGTH_M,
    T2=T2_PS,
    molecular_mass=44e-3/6.023e23  # CO2
)

# AbsorbanceCalculatorの初期化
calculator = AbsorbanceCalculator(
    basis=basis,
    hamiltonian=H0,
    dipole_matrix=dipole_matrix,
    conditions=conditions,
    axes=AXES,
    pol_int=INTERACTION_POL,
    pol_det=DETECTION_POL
)

# 初期密度行列（ボルツマン分布）
rho_init = np.diag(boltz_weights.astype(np.complex128))

# 波数グリッド
wavenumber = np.arange(WN_MIN, WN_MAX + 0.5 * WN_STEP, WN_STEP)

# 各スペクトルの計算
print("Calculating initial state spectrum...")
abs_init = calculator.calculate(rho_init, wavenumber, method='loop')

print("Calculating final state spectrum...")
abs_final = calculator.calculate(rho_final, wavenumber, method='loop')

print("Calculating difference spectrum...")
# 差分密度行列
rho_diff = rho_final - rho_init
# |ΔV|>1成分を零化（振動選択則）
v = basis.basis[:, 0]
dv_mask = (np.abs(v[:, None] - v[None, :]) <= 1)
rho_diff_masked = rho_diff * dv_mask
abs_diff = calculator.calculate(rho_diff_masked, wavenumber, method='loop')

# %% 結果の保存
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 密度行列の保存
rho_path = os.path.join(results_dir, f"rho_final_calculator_{timestamp}.npy")
np.save(rho_path, rho_final)
print(f"\nSaved final density matrix: {rho_path}")

# スペクトルデータの保存
spec_path = os.path.join(results_dir, f"spectra_calculator_{timestamp}.npz")
np.savez_compressed(
    spec_path,
    wavenumber=wavenumber,
    abs_init=abs_init,
    abs_final=abs_final,
    abs_diff=abs_diff,
    rho_init=rho_init,
    rho_final=rho_final,
    temperature=TEMPERATURE_K,
    pressure=PRESSURE_PA,
    T2_ps=T2_PS,
    path_length=PATH_LENGTH_M
)
print(f"Saved spectral data: {spec_path}")

# wavenumber と abs_diff のペアデータを保存
diff_spectrum_data = np.column_stack((wavenumber, abs_diff))
diff_spectrum_path = os.path.join(results_dir, f"wavenumber_abs_diff_pair_{timestamp}.txt")
np.savetxt(
    diff_spectrum_path,
    diff_spectrum_data,
    header="Wavenumber [cm^-1]\tAbsorbance_Difference [mOD]",
    fmt="%.6f\t%.6e",
    delimiter="\t"
)
print(f"Saved wavenumber-abs_diff pair data: {diff_spectrum_path}")

# CSV形式でも保存
diff_spectrum_csv_path = os.path.join(results_dir, f"wavenumber_abs_diff_pair_{timestamp}.csv")
np.savetxt(
    diff_spectrum_csv_path,
    diff_spectrum_data,
    header="wavenumber_cm-1,absorbance_difference_mOD",
    fmt="%.6f,%.6e",
    delimiter=",",
    comments=""
)
print(f"Saved wavenumber-abs_diff pair data (CSV): {diff_spectrum_csv_path}")

# %% 結果のプロット

# Figure 1: 時間発展
fig1, axes1 = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 電場
Efield_data = Efield.get_Efield()
axes1[0].plot(time4Efield, Efield_data[:, 0], "r-", linewidth=1.5, label=r"$E_x(t)$")
axes1[0].plot(time4Efield, Efield_data[:, 1], "b-", linewidth=1.5, label=r"$E_y(t)$")
axes1[0].set_ylabel("Electric Field [V/m]")
axes1[0].set_title(f"Rovibrational Excitation (T={TEMPERATURE_K:.1f} K, E={EFIELD_AMPLITUDE:.2e} V/m)")
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)

# 各状態の占有確率
prob_t = diag_rho_t
total_prob = np.sum(prob_t, axis=1)
main_states = get_plot_states()

for (v, J, M) in main_states:
    if (v, J, M) in basis.index_map:
        idx = basis.get_index((v, J, M))
        axes1[1].plot(time4psi, prob_t[:, idx], linewidth=2, 
                     label=rf"$|{v},{J},{M}\rangle$")

axes1[1].plot(time4psi, total_prob, "k--", alpha=0.7, linewidth=1, label="Total")
axes1[1].set_ylabel("Population")
axes1[1].legend(ncol=2, fontsize=8)
axes1[1].grid(True, alpha=0.3)
axes1[1].set_ylim(-0.05, 1.05)

# 振動準位別の占有確率
vib_probs = {}
for i, (v, J, M) in enumerate(basis.basis):
    if v not in vib_probs:
        vib_probs[v] = np.zeros_like(time4psi)
    vib_probs[v] += prob_t[:, i]

for v in sorted(vib_probs.keys())[:V_MAX+1]:  # 最初の5振動準位
    axes1[2].plot(time4psi, vib_probs[v], linewidth=2, label=rf"$v={v}$")

axes1[2].set_ylabel("Vibrational Population")
axes1[2].legend(ncol=3)
axes1[2].grid(True, alpha=0.3)
axes1[2].set_ylim(-0.05, 1.05)

# 回転準位別の占有確率
rot_probs = {}
for i, (v, J, M) in enumerate(basis.basis):
    if J not in rot_probs:
        rot_probs[J] = np.zeros_like(time4psi)
    rot_probs[J] += prob_t[:, i]

for J in sorted(rot_probs.keys())[:J_MAX+1]:  # 最初の5回転準位
    axes1[3].plot(time4psi, rot_probs[J], linewidth=2, label=rf"$J={J}$")

axes1[3].set_xlabel("Time [fs]")
axes1[3].set_ylabel("Rotational Population")
axes1[3].legend(ncol=3)
axes1[3].grid(True, alpha=0.3)
axes1[3].set_ylim(-0.05, 1.05)

plt.tight_layout()
# %%
# Figure 2: 吸収スペクトル
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 初期状態スペクトル
axes2[0].plot(wavenumber, abs_init, "b-", linewidth=1.5)
axes2[0].set_ylabel("Absorbance [mOD]")
axes2[0].set_title("Initial State (Boltzmann Distribution)")
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim(WN_MIN, WN_MAX)

# 最終状態スペクトル
axes2[1].plot(wavenumber, abs_final, "r-", linewidth=1.5)
axes2[1].set_ylabel("Absorbance [mOD]")
axes2[1].set_title("Final State (After Excitation)")
axes2[1].grid(True, alpha=0.3)

# 差分スペクトル（吸収変化）
axes2[2].plot(wavenumber, abs_diff, "k-", linewidth=1.5)
axes2[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes2[2].set_xlabel("Wavenumber [cm⁻¹]")
axes2[2].set_ylabel("Absorbance Change [mOD]")
axes2[2].set_title("Difference Spectrum (Final - Initial)")
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
# %%
# Figure 3: スペクトル比較（重ね合わせ）
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

# 正規化して比較
abs_init_norm = abs_init / np.max(np.abs(abs_init)) if np.max(np.abs(abs_init)) > 0 else abs_init
abs_final_norm = abs_final / np.max(np.abs(abs_final)) if np.max(np.abs(abs_final)) > 0 else abs_final
abs_diff_norm = abs_diff / np.max(np.abs(abs_diff)) if np.max(np.abs(abs_diff)) > 0 else abs_diff

ax3.plot(wavenumber, abs_init_norm, "b-", linewidth=1.5, alpha=0.7, label="Initial State (Normalized)")
ax3.plot(wavenumber, abs_final_norm, "r-", linewidth=1.5, alpha=0.7, label="Final State (Normalized)")
ax3.plot(wavenumber, abs_diff_norm, "k-", linewidth=2, label="Difference (Normalized)")
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

ax3.set_xlabel("Wavenumber [cm⁻¹]")
ax3.set_ylabel("Normalized Absorbance")
ax3.set_title(f"Absorbance Spectrum Comparison (T={TEMPERATURE_K:.1f} K, P={PRESSURE_PA:.1e} Pa)")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(WN_MIN, WN_MAX)

plt.tight_layout()

# %%
# Figure 4: 吸収変化スペクトルのみ（振動遷移周波数マーク付き）
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['ytick.minor.size'] = 2.5

ylim = (-18, 15) 
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 3))

# 吸収変化スペクトルをプロット
ax4.plot(wavenumber, abs_diff, "k-", linewidth=1, label="Absorbance Change")
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 振動遷移周波数の計算と描画
# V_MAXまでの全ての v→v+1 遷移を計算
transition_freqs = []
transition_labels = []
for v in range(V_MAX):
    # v→v+1 遷移周波数: ω₀₁ - 2χω₀₁(v+1)
    nu_transition = OMEGA_01 - DOMEGA * (v)
    if WN_MIN <= nu_transition <= WN_MAX:  # 表示範囲内のみ
        transition_freqs.append(nu_transition)
        transition_labels.append(f'{v}→{v+1}')

# スペクトルの最大値と最小値を取得してy軸範囲を設定
# y_min, y_max = np.min(abs_diff), np.max(abs_diff)
y_min, y_max = ylim
y_range = y_max - y_min
y_margin = 0.0 * y_range  # マージンを15%に設定
y_min_plot = y_min + y_margin
y_max_plot = y_max - y_margin

# 線の高さを90%に設定
line_height = y_min_plot + 0.92 * (y_max_plot - y_min_plot)

# 一括で振動遷移線を描画
if transition_freqs:  # 遷移周波数がある場合のみ
    ax4.vlines(transition_freqs, y_min_plot, line_height, 
              colors='black', linestyles=':', linewidth=1., alpha=0.5)
    
    # 上部にガイドラベルを追加
    for freq, label in zip(transition_freqs, transition_labels):
        ax4.text(freq, line_height + 0.05 * (y_max_plot - y_min_plot), label, 
                rotation=0, ha='center', va='top', fontsize=9, color='black', alpha=0.5)


# 電場のパワースペクトルを計算してtwinxで追加

# 電場データを取得
Efield_data = Efield.get_Efield()
# x方向とy方向の電場を合成（偏光に応じて）
E_total = Efield_data[:, 0]
len0pd = 2**(int(np.log2(len(E_total)))+4)
E_total_padded = np.pad(E_total, (0, len0pd), mode='constant', constant_values=0)
# 時間軸の設定
dt_s = DT_EFIELD * 1e-15  # fs → s
sampling_freq = 1.0 / dt_s  # Hz
    
# FFTでパワースペクトルを計算
E_fft = np.fft.fft(E_total_padded)
power_spectrum = np.abs(E_fft)**2

# 周波数軸を波数に変換
freqs = np.fft.fftfreq(len(E_total_padded), dt_s)  # Hz
# 正の周波数のみ取得
positive_freq_mask = freqs > 0
freqs_positive = freqs[positive_freq_mask]
power_positive = power_spectrum[positive_freq_mask]

# Hz → cm^-1 変換 (c = 2.998e10 cm/s)
c_cm_per_s = 2.998e10
wavenumbers_power = freqs_positive / c_cm_per_s

# 表示範囲内のデータのみ抽出
power_mask = (wavenumbers_power >= WN_MIN) & (wavenumbers_power <= WN_MAX)
wavenumbers_power_plot = wavenumbers_power[power_mask]
power_plot = power_positive[power_mask]

# パワースペクトルを正規化
if len(power_plot) > 0 and np.max(power_plot) > 0:
    power_plot_normalized = power_plot / np.max(power_plot) * np.max(abs_diff)*0.7  
    
    # パワースペクトルをプロット
    ax4.plot(wavenumbers_power_plot, power_plot_normalized, 'gray', 
                 alpha=0.6, linewidth=1, label='Power Spectrum (normalized)')
    ax4.fill_between(wavenumbers_power_plot, power_plot_normalized, color='gray', alpha=0.1)
    ax4.text(0.55, 0.75, 'Power Spectrum (normalized)', color='gray', fontsize=10, transform=ax4.transAxes)
    
# 軸の設定
ax4.set_xlabel("Wavenumber (cm⁻¹)")
ax4.set_ylabel("Absorbance Change (mOD)")
# ax4.set_title(f"Absorption Change Spectrum with Vibrational Transitions\n(T={TEMPERATURE_K:.1f} K, Fluence={FLUENCE_MJ_CM2:.1f} mJ/cm²)")
# ax4.legend(loc='best')
ax4.grid(False)
ax4.set_xlim(WN_MIN, WN_MAX)
ax4.set_ylim(ylim)
ax4.invert_xaxis()

# 最大変化の位置を表示
# max_change_idx = np.argmax(np.abs(abs_diff))
# max_change_wn = wavenumber[max_change_idx]
# max_change_val = abs_diff[max_change_idx]
# ax4.annotate(f'Max change: {max_change_val:.2f} mOD\nat {max_change_wn:.1f} cm⁻¹',
#             xy=(max_change_wn, max_change_val),
#             xytext=(max_change_wn + 50, max_change_val + 5),
#             arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
#             fontsize=10, ha='left')

plt.tight_layout()

# Figure 4用データの保存
fig4_absorbance_data = np.column_stack((wavenumber, abs_diff))
fig4_absorbance_path = os.path.join(results_dir, f"fig4_absorbance_data_{timestamp}.npy")
np.save(fig4_absorbance_path, fig4_absorbance_data)
print(f"Saved Figure 4 absorbance data: {fig4_absorbance_path}")

if len(power_plot) > 0 and np.max(power_plot) > 0:
    fig4_power_data = np.column_stack((wavenumbers_power_plot, power_plot_normalized))
    fig4_power_path = os.path.join(results_dir, f"fig4_power_data_{timestamp}.npy")
    np.save(fig4_power_path, fig4_power_data)
    print(f"Saved Figure 4 power spectrum data: {fig4_power_path}")

# 計算設定パラメータの保存
settings_path = os.path.join(results_dir, f"calculation_settings_{timestamp}.txt")
with open(settings_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("CALCULATION SETTINGS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    
    f.write("# System Parameters\n")
    f.write(f"V_MAX = {V_MAX}  # Maximum vibrational quantum number\n")
    f.write(f"J_MAX = {J_MAX}  # Maximum rotational quantum number\n")
    f.write(f"USE_M = {USE_M}  # Use magnetic quantum number\n")
    f.write(f"Basis dimension = {basis.size()}\n\n")
    
    f.write("# Molecular Parameters (CO2)\n")
    f.write(f"OMEGA_01 = {OMEGA_01}  # Vibrational frequency [cm^-1]\n")
    f.write(f"DOMEGA = {DOMEGA}  # Anharmonicity correction [cm^-1]\n")
    f.write(f"B_CONSTANT = {B_CONSTANT}  # Rotational constant [cm^-1]\n")
    f.write(f"ALPHA_CONSTANT = {ALPHA_CONSTANT}  # Vibration-rotation interaction [cm^-1]\n")
    f.write(f"MU0 = {MU0}  # Dipole matrix element magnitude [C·m]\n\n")
    
    f.write("# Temperature Settings\n")
    f.write(f"TEMPERATURE_K = {TEMPERATURE_K}  # Temperature [K]\n")
    f.write(f"BOLTZMANN_WEIGHT_THRESHOLD = {BOLTZMANN_WEIGHT_THRESHOLD}  # Weight threshold\n")
    f.write(f"Selected states = {len(nonzero_indices)} / {basis.size()}\n\n")
    
    f.write("# Laser Pulse Settings\n")
    f.write(f"PULSE_DURATION_FWHM = {PULSE_DURATION_FWHM}  # Pulse width FWHM [fs]\n")
    f.write(f"PULSE_DURATION = {PULSE_DURATION:.3f}  # Pulse width (1/e) [fs]\n")
    f.write(f"GDD = {GDD}  # Group delay dispersion [fs^2]\n")
    f.write(f"FLUENCE_MJ_CM2 = {FLUENCE_MJ_CM2}  # Fluence [mJ/cm^2]\n")
    f.write(f"EFIELD_AMPLITUDE = {EFIELD_AMPLITUDE:.3e}  # Electric field amplitude [V/m]\n")
    f.write(f"POLARIZATION = {POLARIZATION.tolist()}  # Polarization direction\n")
    f.write(f"AXES = '{AXES}'  # Dipole axes considered\n")
    f.write(f"CARRIER_STATE_LOWER = {CARRIER_STATE_LOWER}  # Lower state (v,J,M)\n")
    f.write(f"CARRIER_STATE_UPPER = {CARRIER_STATE_UPPER}  # Upper state (v,J,M)\n")
    f.write(f"Carrier frequency = {float(carrier_freq):.1f} cm^-1\n\n")
    
    f.write("# Time Grid Settings\n")
    f.write(f"TIME_START = {TIME_START}  # Start time [fs]\n")
    f.write(f"TIME_END = {TIME_END}  # End time [fs]\n")
    f.write(f"DT_EFIELD = {DT_EFIELD}  # Electric field sampling interval [fs]\n")
    f.write(f"SAMPLE_STRIDE = {SAMPLE_STRIDE}  # Sampling stride\n\n")
    
    f.write("# Spectroscopy Settings\n")
    f.write(f"TEMPERATURE_SPEC = {TEMPERATURE_SPEC}  # Spectroscopy temperature [K]\n")
    f.write(f"PRESSURE_PA = {PRESSURE_PA}  # Pressure [Pa]\n")
    f.write(f"PATH_LENGTH_M = {PATH_LENGTH_M}  # Optical path length [m]\n")
    f.write(f"T2_PS = {T2_PS}  # Coherence relaxation time [ps]\n")
    f.write(f"INTERACTION_POL = {INTERACTION_POL.tolist()}  # Interaction polarization\n")
    f.write(f"DETECTION_POL = {DETECTION_POL.tolist()}  # Detection polarization\n\n")
    
    f.write("# Spectrum Range\n")
    f.write(f"WN_MIN = {WN_MIN}  # Minimum wavenumber [cm^-1]\n")
    f.write(f"WN_MAX = {WN_MAX}  # Maximum wavenumber [cm^-1]\n")
    f.write(f"WN_STEP = {WN_STEP}  # Wavenumber step [cm^-1]\n")
    f.write(f"Total spectrum points = {len(wavenumber)}\n\n")
    
    f.write("# Parallel Processing\n")
    f.write(f"USE_PARALLEL = {USE_PARALLEL}\n")
    f.write(f"PARALLEL_BACKEND = '{PARALLEL_BACKEND}'\n")
    f.write(f"CHUNK_SIZE = {CHUNK_SIZE}\n")
    f.write(f"MAX_WORKERS = {MAX_WORKERS}\n\n")
    
    f.write("# Vibrational Transition Frequencies\n")
    for v in range(min(V_MAX, 10)):  # 最初の10遷移まで
        nu_trans = OMEGA_01 - DOMEGA * v
        f.write(f"v={v}→{v+1}: {nu_trans:.1f} cm^-1\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("CALCULATION RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Time evolution elapsed time: {end - start:.3f} s\n")
    f.write(f"Initial spectrum maximum: {np.max(abs_init):.2f} mOD\n")
    f.write(f"Final spectrum maximum: {np.max(abs_final):.2f} mOD\n")
    f.write(f"Maximum difference spectrum change: {np.max(np.abs(abs_diff)):.2f} mOD\n")
    f.write(f"Partition function Z: {Z_value:.6e}\n")
    f.write(f"Kept Boltzmann weight: {kept_weight:.3e}\n")

print(f"Saved calculation settings: {settings_path}")

# %%
# 図の保存
for i, (fig, name) in enumerate([(fig1, "dynamics"), (fig2, "spectra"), (fig3, "comparison"), (fig4, "absorbance_change")], 1):
    fig_path = os.path.join(results_dir, f"figure{i}_{name}_calculator_{timestamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure {i}: {fig_path}")

plt.show()

# %% サマリーの出力
print("\n" + "=" * 70)
print("Simulation Complete")
print("=" * 70)
print(f"Temperature: {TEMPERATURE_K:.1f} K")
print(f"Pressure: {PRESSURE_PA:.2e} Pa")
print(f"Path length: {PATH_LENGTH_M*1000:.1f} mm")
print(f"Relaxation time T2: {T2_PS:.1f} ps")
print(f"Pulse duration: {PULSE_DURATION:.1f} fs")
print(f"Fluence: {FLUENCE_MJ_CM2:.1f} mJ/cm²")
print(f"Electric field amplitude: {EFIELD_AMPLITUDE:.2e} V/m")
print(f"\nInitial spectrum maximum: {np.max(abs_init):.2f} mOD")
print(f"Final spectrum maximum: {np.max(abs_final):.2f} mOD")
print(f"Maximum difference spectrum change: {np.max(np.abs(abs_diff)):.2f} mOD")
print("=" * 70)
