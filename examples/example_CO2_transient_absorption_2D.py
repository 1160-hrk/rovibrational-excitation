#!/usr/bin/env python
"""
二次元過渡吸収スペクトル計算：遅延時間掃引
====================================================

ボルツマン分布の初期状態から振動回転励起後、密度行列を遅延時間だけ
固有ハミルトニアンで時間発展させ、各遅延時間での吸収変化スペクトルを
計算することで二次元マップを作成。

機能:
- レーザーによる振動回転励起をボルツマン分布で重ね合わせ
- 密度行列を時間発展してrho_initを引き吸収変化スペクトルを計算  
- 吸収変化スペクトルの遅延時間依存性を二次元マップでプロット

実行方法:
    python examples/example_CO2_transient_absorption_2D.py
"""

import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
CHUNK_SIZE = 8
MAX_WORKERS = min(os.cpu_count() or 1, 12)
PROGRESS_EVERY = 1  # 進捗ログ出力のバッチ間隔

# ===== システムパラメータ =====
SPARSE = True
DENSE = False
V_MAX = 8  # 最大振動量子数
J_MAX = 12  # 最大回転量子数
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
GDD = 0.0  # [fs^2]
POLARIZATION = np.array([1, 0])  # x方向偏光
AXES = "xy"  # x, y方向の双極子を考慮

# キャリア周波数を設定するための状態ペア（下位, 上位）
CARRIER_STATE_LOWER = (2, 0, 0)
CARRIER_STATE_UPPER = (3, 1, 1)

# ===== フルエンス設定 =====
FLUENCE_MJ_CM2 = 100.0  # [mJ/cm^2]
F_SI = FLUENCE_MJ_CM2 * 10.0  # [J/m^2]

# ===== 時間グリッド設定 =====
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = PULSE_DURATION * 10  # 終了時間 [fs]
DT_EFIELD = 0.05  # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 1  # サンプリングストライド

# ===== 遅延時間設定 =====
DELAY_MIN = 0.0  # 最小遅延時間 [fs]
DELAY_MAX = 22000.0  # 最大遅延時間 [fs]
DELAY_STEP = 250.0  # 遅延時間ステップ [fs]
DELAY_TIMES = np.arange(DELAY_MIN, DELAY_MAX + 0.5 * DELAY_STEP, DELAY_STEP)

# ===== 吸収スペクトル設定 =====
# 実験条件
TEMPERATURE_SPEC = TEMPERATURE_K  # スペクトル計算時の温度 [K]
PRESSURE_PA = 6.0e-1  # 圧力 [Pa]
PATH_LENGTH_M = 1.0e-3  # 光路長 [m]
# T2_PS = 667.0  # コヒーレンス緩和時間 [ps]
T2_PS = 100.0  # コヒーレンス緩和時間 [ps]

# 偏光設定
INTERACTION_POL = np.array([np.sqrt(1/3), np.sqrt(2/3)])  # 相互作用光の偏光
# INTERACTION_POL = np.array([1.0, 0.0])  # 相互作用光の偏光
# DETECTION_POL = np.array([np.sqrt(1/3), np.sqrt(2/3)])  # マジックアングル検出 54.7°
DETECTION_POL = INTERACTION_POL  # 相互作用光と同じ偏光

# スペクトル範囲（計算時間のため範囲を狭める）
WN_MIN, WN_MAX, WN_STEP = 2100.0, 2400.0, 0.02  # [cm^-1]

# ===== プロット設定 =====
def get_plot_states():
    """Select main states for plotting"""
    states = []
    for V in range(min(V_MAX+1, 3)):  # First 3 vibrational levels
        for J in range(min(J_MAX+1, 3)):  # First 3 rotational levels
            for M in range(-J, J+1):
                if AXES == "xy":
                    if (V + J) % 2 == 0 and (J + M) % 2 == 0:
                        states.append((V, J, M))
                elif AXES == "zx":
                    if (V + J) % 2 == 0 and M == 0:
                        states.append((V, J, M))
                else:
                    raise ValueError(f"Invalid axes: {AXES}")
    return states[:8]  # Maximum 8 states


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


def evolve_density_matrix(rho: np.ndarray, hamiltonian_matrix: np.ndarray, time_fs: float) -> np.ndarray:
    """
    密度行列を対角ハミルトニアンで時間発展させる
    
    Parameters
    ----------
    rho : np.ndarray
        初期密度行列
    hamiltonian_matrix : np.ndarray
        対角ハミルトニアン行列 [J] (対角要素が固有値)
    time_fs : float
        発展時間 [fs]
        
    Returns
    -------
    np.ndarray
        時間発展後の密度行列
    """
    # 時間をSI単位に変換
    time_s = time_fs * 1e-15
    
    # ハミルトニアンは既に対角なので対角要素（固有値）を取得
    eigenvalues = np.diag(hamiltonian_matrix)
    
    # 時間発展演算子 U(t) = exp(-i H t / ℏ)
    HBAR_SI = CONSTANTS.HBAR  # [J*s]
    U = np.diag(np.exp(-1j * eigenvalues * time_s / HBAR_SI))
    U_dag = U.conj().T  # U†
    
    # 密度行列の時間発展: ρ(t) = U(t) ρ(0) U†(t)
    rho_evolved = U @ rho @ U_dag
    
    return rho_evolved


# %% 基底・ハミルトニアン・双極子行列の生成
print("=" * 70)
print("2D Transient Absorption Spectroscopy with Delay Time Scanning")
print("=" * 70)
print(f"Basis size: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")
print(f"Delay times: {DELAY_MIN:.0f} - {DELAY_MAX:.0f} fs, step = {DELAY_STEP:.0f} fs")
print(f"Number of delay points: {len(DELAY_TIMES)}")

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
hamiltonian_matrix = H0.get_matrix(units="J")
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

# %% 時間発展計算（励起直後の密度行列作成）

print("\nStarting pump excitation calculation...")
start = time.perf_counter()
diag_rho_t, rho_pump = run_boltzmann_ensemble_parallel(
    nonzero_indices=nonzero_indices,
    boltz_weights=boltz_weights,
    basis_size=basis.size(),
    chunk_size=CHUNK_SIZE,
    max_workers=MAX_WORKERS,
    backend=PARALLEL_BACKEND,
    progress_every=PROGRESS_EVERY,
)
end = time.perf_counter()
print(f"Pump excitation completed. Elapsed time: {end - start:.3f} s")

# %% AbsorbanceCalculatorの設定

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
print(f"Wavenumber range: {WN_MIN:.1f} - {WN_MAX:.1f} cm^-1, {len(wavenumber)} points")

# %% 遅延時間掃引による二次元マップ計算

print(f"\n=== 2D Transient Absorption Map Calculation ===")
print(f"Calculating spectra for {len(DELAY_TIMES)} delay times...")

# 結果格納用配列
abs_diff_2d = np.zeros((len(DELAY_TIMES), len(wavenumber)))
rho_evolved_list = np.zeros((len(DELAY_TIMES), basis.size(), basis.size()), dtype=np.complex128)

# 各遅延時間での計算
for i, delay_time in enumerate(DELAY_TIMES):
    print(f"Processing delay time {delay_time:.1f} fs ({i+1}/{len(DELAY_TIMES)})")
    
    # 密度行列を遅延時間だけ時間発展
    if delay_time == 0.0:
        rho_evolved = rho_pump.copy()
    else:
        rho_evolved = evolve_density_matrix(
            rho_pump, hamiltonian_matrix, float(delay_time)
            )
    
    rho_evolved_list[i, :, :] = rho_evolved
    
    # 差分密度行列の計算
    rho_diff = rho_evolved - rho_init
    
    # 振動選択則マスクの適用（|ΔV|≤1）
    v = basis.basis[:, 0]
    dv_mask = (np.abs(v[:, None] - v[None, :]) <= 1)
    rho_diff_masked = rho_diff * dv_mask
    
    # 吸収変化スペクトルの計算
    abs_diff = calculator.calculate(rho_diff_masked, wavenumber, method='loop')
    abs_diff_2d[i, :] = abs_diff
    
    if (i + 1) % 5 == 0:
        print(f"  Completed {i+1}/{len(DELAY_TIMES)} delay times")

print("2D map calculation completed!")

# %% 結果の保存
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 2Dマップデータの保存
map_path = os.path.join(results_dir, f"transient_absorption_2d_{timestamp}.npz")
np.savez_compressed(
    map_path,
    delay_times=DELAY_TIMES,
    wavenumber=wavenumber,
    abs_diff_2d=abs_diff_2d,
    rho_init=rho_init,
    rho_pump=rho_pump,
    temperature=TEMPERATURE_K,
    pressure=PRESSURE_PA,
    T2_ps=T2_PS,
    path_length=PATH_LENGTH_M,
    fluence=FLUENCE_MJ_CM2
)
print(f"\nSaved 2D map data: {map_path}")

# 設定パラメータの保存
settings_path = os.path.join(results_dir, f"transient_2d_settings_{timestamp}.txt")
with open(settings_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("2D TRANSIENT ABSORPTION CALCULATION SETTINGS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    
    f.write("# System Parameters\n")
    f.write(f"V_MAX = {V_MAX}  # Maximum vibrational quantum number\n")
    f.write(f"J_MAX = {J_MAX}  # Maximum rotational quantum number\n")
    f.write(f"USE_M = {USE_M}  # Use magnetic quantum number\n")
    f.write(f"Basis dimension = {basis.size()}\n\n")
    
    f.write("# Delay Time Settings\n")
    f.write(f"DELAY_MIN = {DELAY_MIN}  # Minimum delay time [fs]\n")
    f.write(f"DELAY_MAX = {DELAY_MAX}  # Maximum delay time [fs]\n")
    f.write(f"DELAY_STEP = {DELAY_STEP}  # Delay time step [fs]\n")
    f.write(f"Number of delay points = {len(DELAY_TIMES)}\n\n")
    
    f.write("# Spectrum Range\n")
    f.write(f"WN_MIN = {WN_MIN}  # Minimum wavenumber [cm^-1]\n")
    f.write(f"WN_MAX = {WN_MAX}  # Maximum wavenumber [cm^-1]\n")
    f.write(f"WN_STEP = {WN_STEP}  # Wavenumber step [cm^-1]\n")
    f.write(f"Total spectrum points = {len(wavenumber)}\n\n")
    
    f.write("# Calculation Results\n")
    f.write(f"Maximum absolute change: {np.max(np.abs(abs_diff_2d)):.2f} mOD\n")
    f.write(f"Minimum change: {np.min(abs_diff_2d):.2f} mOD\n")
    f.write(f"Maximum change: {np.max(abs_diff_2d):.2f} mOD\n")

print(f"Saved calculation settings: {settings_path}")

# %% 結果のプロット

# Figure 1: 2D過渡吸収マップ
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))


# 2Dカラーマップ
contour_levels = 1000
vmax = np.max(np.abs(abs_diff_2d))*0.07
vmin = -vmax
X, Y = np.meshgrid(wavenumber, DELAY_TIMES/1000)

# コンターを使用してスムーズな表示
# map2d = ax1.contourf(X, Y, abs_diff_2d, levels=contour_levels, 
#                       cmap='RdBu_r', vmin=vmin, vmax=vmax, extend='both')
map2d = ax1.pcolormesh(X, Y, abs_diff_2d, 
                       cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')

# カラーバーの追加（inset_axesを使用して位置調整）

cax = inset_axes(ax1, width="1.5%", height="30%", loc='lower right', 
                bbox_to_anchor=(0.05, -0.01, 1, 1), bbox_transform=ax1.transAxes)
cbar = plt.colorbar(map2d, cax=cax, label=r'$\Delta A$ (mOD)', extend='both')
cbar.mappable.set_clim(vmin, vmax)
cbar.ax.tick_params(labelsize=10)

# 振動遷移周波数の線を描画
for v in range(V_MAX):  # 最初の5遷移
    nu_transition = OMEGA_01 - DOMEGA * v
    if WN_MIN <= nu_transition <= WN_MAX:
        ax1.axvline(nu_transition, ymin=0, ymax=0.92, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(nu_transition, DELAY_MAX/1000 * 0.99, f'{v}→{v+1}', 
                rotation=0, ha='center', va='top', color='black', fontsize=9, alpha=0.5)

ax1.set_xlabel(r'Wavenumber (cm$^{-1}$)')
ax1.set_ylabel(r'Delay Time (ps)')
ax1.set_title(f'2D Transient Absorption Map(T={TEMPERATURE_K:.1f} K, Fluence={FLUENCE_MJ_CM2:.1f} mJ/cm²)')
ax1.invert_xaxis()

plt.tight_layout()
# %%
# Figure 2: 特定遅延時間でのスペクトル
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# いくつかの代表的な遅延時間をプロット
plot_delays = [0, len(DELAY_TIMES)//4, len(DELAY_TIMES)//2, 3*len(DELAY_TIMES)//4, len(DELAY_TIMES)-1]
colors = ['blue', 'green', 'orange', 'red', 'purple']

for i, color in zip(plot_delays, colors):
    if i < len(DELAY_TIMES):
        ax2.plot(wavenumber, abs_diff_2d[i, :], color=color, linewidth=1.5,
                label=f'{DELAY_TIMES[i]:.0f} fs')

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Wavenumber [cm⁻¹]')
ax2.set_ylabel('Absorbance Change [mOD]')
ax2.set_title('Transient Absorption Spectra at Different Delay Times')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()

# Figure 3: 特定波数での時間依存性
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

# いくつかの代表的な波数をプロット
plot_wavenumbers = [
    OMEGA_01,  # v=0→1遷移
    OMEGA_01 - DOMEGA,  # v=1→2遷移
    OMEGA_01 - 2*DOMEGA,  # v=2→3遷移
]
colors = ['red', 'blue', 'green']

for wn, color in zip(plot_wavenumbers, colors):
    if WN_MIN <= wn <= WN_MAX:
        # 最も近い波数インデックスを見つける
        wn_idx = np.argmin(np.abs(wavenumber - wn))
        actual_wn = wavenumber[wn_idx]
        ax3.plot(DELAY_TIMES, abs_diff_2d[:, wn_idx], color=color, linewidth=2,
                label=f'{actual_wn:.1f} cm⁻¹')

ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Delay Time [fs]')
ax3.set_ylabel('Absorbance Change [mOD]')
ax3.set_title('Time Evolution at Specific Wavenumbers')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Figure 4: 初期スペクトル比較
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))

# t=0での差分スペクトルとt=最大での差分スペクトルを比較
ax4.plot(wavenumber, abs_diff_2d[0, :], 'b-', linewidth=1.5, 
         label=f't = {DELAY_TIMES[0]:.0f} fs')
ax4.plot(wavenumber, abs_diff_2d[-1, :], 'r-', linewidth=1.5, 
         label=f't = {DELAY_TIMES[-1]:.0f} fs')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax4.set_xlabel('Wavenumber [cm⁻¹]')
ax4.set_ylabel('Absorbance Change [mOD]')
ax4.set_title('Comparison: Initial vs Final Delay Time Spectra')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.invert_xaxis()

plt.tight_layout()

# 図の保存
for i, (fig, name) in enumerate([(fig1, "2d_map"), (fig2, "delay_spectra"), 
                                 (fig3, "time_evolution"), (fig4, "comparison")], 1):
    fig_path = os.path.join(results_dir, f"figure{i}_{name}_transient2d_{timestamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure {i}: {fig_path}")

plt.show()

# %% サマリーの出力
print("\n" + "=" * 70)
print("2D Transient Absorption Simulation Complete")
print("=" * 70)
print(f"Temperature: {TEMPERATURE_K:.1f} K")
print(f"Basis dimension: {basis.size()}")
print(f"Delay time range: {DELAY_MIN:.0f} - {DELAY_MAX:.0f} fs")
print(f"Number of delay points: {len(DELAY_TIMES)}")
print(f"Wavenumber range: {WN_MIN:.1f} - {WN_MAX:.1f} cm⁻¹")
print(f"Number of wavenumber points: {len(wavenumber)}")
print(f"Maximum absolute change: {np.max(np.abs(abs_diff_2d)):.2f} mOD")
print(f"Fluence: {FLUENCE_MJ_CM2:.1f} mJ/cm²")
print(f"Electric field amplitude: {EFIELD_AMPLITUDE:.2e} V/m")
print("=" * 70)
