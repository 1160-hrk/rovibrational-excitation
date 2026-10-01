#!/usr/bin/env python
"""
GRAPEによる振動回転励起最適制御
=================================

線形分子の振動回転準位間で高フィデリティのポピュレーション移行を実現する
電場波形をGRAPE（Gradient Ascent Pulse Engineering）で最適化。

実行方法:
    python examples/grape_rovibrational_optimization.py

参考:
- S. Machnes et al., Phys. Rev. Lett. 120, 053203 (2018)
- D. M. Reich et al., J. Chem. Phys. 136, 104103 (2012)
"""

import os
import sys
import time
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis, StateVector
from rovibrational_excitation.fields import ElectricField, gaussian
from rovibrational_excitation.dynamics import SchrodingerPropagator
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.dynamics.utils import cm_to_rad_phz


# =========================
# Parameters
# =========================
# System parameters
V_MAX = 1
J_MAX = 1
USE_M = True

# Molecular parameters
OMEGA_01 = 2349.1  # [cm^-1]
DOMEGA = 25        # [cm^-1]
B_CONSTANT = 0.39  # [cm^-1]
ALPHA_CONSTANT = 0.0037  # [cm^-1]
UNIT_FREQUENCY = "cm^-1"
MU0 = 1e-30  # [C*m]
UNIT_DIPOLE = "C*m"

# Initial/Target
INITIAL_STATE = (0, 0, 0)
TARGET_STATE = (1, 1, 1)

# Time grid
TIME_TOTAL = 1000.0  # [fs]
DT_EFIELD = 0.5      # [fs]
SAMPLE_STRIDE = 1

# GRAPE optimization parameters
MAX_ITER = 1000
TARGET_FIDELITY = 1.0
LEARNING_RATE = 5e18  # 電場更新の学習率（単位整合済みの小さめ値）
LAMBDA_A = 1e-19       # 電場二乗ペナルティの係数
CONVERGENCE_TOL = 1e-20
MIN_ITER_EARLYSTOP = 50
EARLYSTOP_MIN_FIDELITY = 0.90

# Field constraints
FIELD_MAX = 1e9  # [V/m]
ENFORCE_HARD_CLIP = False

# Envelope (Gaussian FWHM) with optional sin^2 window for zero boundaries
ENVELOPE_DURATION = None  # e.g., TIME_TOTAL*4 で幅広ガウシアン
USE_SIN2_ENVELOPE = False


# =========================
# Utilities
# =========================
def calculate_fidelity(psi_final: np.ndarray, target_idx: int) -> float:
    return float(np.abs(psi_final[target_idx]) ** 2)


def shape_function(t: np.ndarray, T: float) -> np.ndarray:
    return np.sin(np.pi * t / T) ** 2


def gaussian_envelope_fwhm(t: np.ndarray, t_center: float, fwhm: Optional[float]) -> np.ndarray:
    if fwhm is None or fwhm <= 0:
        return np.ones_like(t)
    return np.exp(-4.0 * np.log(2.0) * ((t - t_center) / float(fwhm)) ** 2)


def windowed_gaussian_envelope(t: np.ndarray, T: float, t_center: float,
                               duration_fwhm: Optional[float], use_sin2: bool = True) -> np.ndarray:
    g = gaussian_envelope_fwhm(t, t_center, duration_fwhm)
    if use_sin2:
        return g * shape_function(t, T)
    return g


def apply_envelope_to_field(t: np.ndarray, field_data: np.ndarray, envelope: Optional[np.ndarray]) -> np.ndarray:
    if envelope is None:
        return field_data
    return field_data * envelope[:, None]


# =========================
# GRAPE Optimizer
# =========================
class GRAPEOptimizer:
    """GRAPE（Gradient Ascent Pulse Engineering）を用いた最適化クラス"""

    def __init__(self, basis: LinMolBasis, hamiltonian, dipole_matrix: LinMolDipoleMatrix,
                 initial_idx: int, target_idx: int, time_total: float, dt: float,
                 sample_stride: int = 1) -> None:
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.dipole_matrix = dipole_matrix
        self.initial_idx = initial_idx
        self.target_idx = target_idx
        self.dt = float(dt)
        self.sample_stride = sample_stride

        # Time grid consistent with RK4 scheme used in the propagator
        target_traj_steps = int(time_total / dt) + 1
        required_field_steps = 2 * (target_traj_steps - 1) + 1
        self.tlist = np.linspace(0.0, time_total, required_field_steps)
        self.n_steps = len(self.tlist)
        self.n_traj_steps = target_traj_steps

        # Propagator
        self.propagator = SchrodingerPropagator(
            backend="numpy", validate_units=True, renorm=True
        )

        # States
        self.psi_initial = np.zeros(basis.size(), dtype=complex)
        self.psi_initial[initial_idx] = 1.0
        self.psi_target = np.zeros(basis.size(), dtype=complex)
        self.psi_target[target_idx] = 1.0

        # Dipole in propagation units (same convention as the propagator expects)
        mu_x_si = self.dipole_matrix.get_mu_x_SI()
        mu_y_si = self.dipole_matrix.get_mu_y_SI()
        if hasattr(mu_x_si, "toarray"):
            mu_x_si = mu_x_si.toarray()
        if hasattr(mu_y_si, "toarray"):
            mu_y_si = mu_y_si.toarray()
        # Use same converter as the reference (μ' consistent with H units)
        self.mu_x_prime = cm_to_rad_phz(mu_x_si)
        self.mu_y_prime = cm_to_rad_phz(mu_y_si)

        # Envelope window
        self.envelope_window = None

        print("初期化完了:")
        print(f"  基底次元: {basis.size()}")
        print(f"  電場時間ステップ数: {self.n_steps}")
        print(f"  予想プロパゲーション軌跡長: {self.n_traj_steps}")
        print(f"  サンプリングストライド: {sample_stride}")
        print(f"  初期状態: {basis.basis[initial_idx]} (index={initial_idx})")
        print(f"  目標状態: {basis.basis[target_idx]} (index={target_idx})")

    def forward_propagation(self, efield: ElectricField) -> Tuple[np.ndarray, np.ndarray]:
        result = self.propagator.propagate(
            hamiltonian=self.hamiltonian,
            efield=efield,
            dipole_matrix=self.dipole_matrix,
            initial_state=self.psi_initial,
            axes="xy",
            return_traj=True,
            return_time_psi=True,
            sample_stride=self.sample_stride,
            algorithm="rk4",
            sparse=False,
        )
        return result[0], result[1]

    def backward_propagation(self, efield: ElectricField, psi_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Terminal condition: chi(T) = <phi|psi(T)> |phi>
        psi_final = psi_traj[-1]
        overlap = np.vdot(self.psi_target, psi_final)
        chi_initial = overlap * self.psi_target

        # Reverse-time field
        efield_backward = ElectricField(tlist=self.tlist[::-1])
        efield_backward.add_arbitrary_Efield(efield.get_Efield()[::-1])

        result = self.propagator.propagate(
            hamiltonian=self.hamiltonian,
            efield=efield_backward,
            dipole_matrix=self.dipole_matrix,
            initial_state=chi_initial,
            axes="xy",
            return_traj=True,
            return_time_psi=True,
            sample_stride=self.sample_stride,
            algorithm="rk4",
            sparse=False,
        )

        time_backward = -result[0][::-1]
        chi_backward = result[1][::-1]
        return time_backward, chi_backward

    def compute_gradient_local(self, psi_traj: np.ndarray, chi_traj: np.ndarray,
                               field_data: np.ndarray, lambda_a: float,
                               tlist: np.ndarray) -> np.ndarray:
        """
        時間局所の勾配を用いてGRAPE更新の勾配を計算。
        伝播器の内部離散に合わせ、軌跡インデックス i に対し電場の j=2*i を対応させる。

        勾配: dJ/dE ~ -2*Im<chi|mu|psi>*S(t)*dt + lambda_a*S(t)*E*dt
        （J=1-Fidelity+J_a を最小化。勾配降下方向を採用）
        """
        mu_x = self.mu_x_prime
        mu_y = self.mu_y_prime
        n_traj_steps = len(psi_traj)
        n_field_steps = len(tlist)
        dt_field = float(tlist[1] - tlist[0])
        S_t = shape_function(tlist, tlist[-1])

        grad = np.zeros_like(field_data)
        for i in range(n_traj_steps):
            jf = i * 2
            if jf >= n_field_steps:
                break
            psi_i = psi_traj[i]
            chi_i = chi_traj[i]
            im_x = float(np.imag(np.vdot(chi_i, (mu_x @ psi_i))))
            im_y = float(np.imag(np.vdot(chi_i, (mu_y @ psi_i))))
            S = float(S_t[jf])
            # Terminal cost gradient part (aiming to increase fidelity => decrease J)
            grad_term_x = -2.0 * im_x * S * dt_field
            grad_term_y = -2.0 * im_y * S * dt_field
            # Running cost gradient (lambda_a * S * E * dt)
            grad_run_x = float(lambda_a) * S * field_data[jf, 0] * dt_field
            grad_run_y = float(lambda_a) * S * field_data[jf, 1] * dt_field
            grad[jf, 0] += grad_term_x + grad_run_x
            grad[jf, 1] += grad_term_y + grad_run_y
            if jf + 1 < n_field_steps:
                grad[jf + 1, 0] += grad_term_x + grad_run_x
                grad[jf + 1, 1] += grad_term_y + grad_run_y
        return grad

    def optimize(self, learning_rate: float = LEARNING_RATE, lambda_a: float = LAMBDA_A,
                 max_iter: int = MAX_ITER, convergence_tol: float = CONVERGENCE_TOL,
                 target_fidelity: float = TARGET_FIDELITY) -> Tuple[ElectricField, list, list]:
        print("\nGRAPE最適化開始:")
        print(f"  最大反復回数: {max_iter}")
        print(f"  学習率: {learning_rate:.3e}")
        print(f"  目標フィデリティー: {target_fidelity}")
        print(f"  収束閾値: {convergence_tol}")
        print(f"  ペナルティλ: {lambda_a}")

        # Initial field with x and y components (near-resonant carriers)
        efield = ElectricField(tlist=self.tlist)
        tc = self.tlist[-1] / 2.0

        eigenvalues = self.hamiltonian.get_eigenvalues()
        carrier_freq = float(converter.convert_frequency(
            eigenvalues[self.target_idx] - eigenvalues[self.initial_idx],
            "rad/fs", UNIT_FREQUENCY
        ))

        initial_amplitude = 5e9  # [V/m]
        efield.add_dispersed_Efield(
            envelope_func=gaussian,
            duration=TIME_TOTAL / 2.0,
            t_center=tc,
            carrier_freq=carrier_freq,
            carrier_freq_units=UNIT_FREQUENCY,
            amplitude=initial_amplitude,
            polarization=np.array([1, 0]),
            const_polarisation=True,
        )
        efield.add_dispersed_Efield(
            envelope_func=gaussian,
            duration=TIME_TOTAL / 2.0,
            t_center=tc,
            carrier_freq=carrier_freq,
            carrier_freq_units=UNIT_FREQUENCY,
            amplitude=initial_amplitude,
            polarization=np.array([0, 1]),
            const_polarisation=True,
        )

        # Envelope multiplicatively applied to guarantee zeros at boundaries if desired
        if ENVELOPE_DURATION is not None and ENVELOPE_DURATION > 0:
            self.envelope_window = windowed_gaussian_envelope(
                self.tlist, self.tlist[-1], tc, ENVELOPE_DURATION, USE_SIN2_ENVELOPE
            )
            field_data_init = apply_envelope_to_field(self.tlist, efield.get_Efield(), self.envelope_window)
            efield = ElectricField(tlist=self.tlist)
            efield.add_arbitrary_Efield(field_data_init)
        else:
            self.envelope_window = np.ones_like(self.tlist)

        print(f"  初期キャリア周波数: {carrier_freq:.3f} {UNIT_FREQUENCY}")
        print(f"  初期振幅: {initial_amplitude:.3e} V/m")

        # Histories
        fidelity_history: list[float] = []
        field_norm_history: list[float] = []
        terminal_cost_history: list[float] = []
        running_cost_history: list[float] = []
        total_cost_history: list[float] = []

        start_time = time.time()
        dt_field = float(self.tlist[1] - self.tlist[0])
        S_t = shape_function(self.tlist, self.tlist[-1])

        for iteration in range(max_iter):
            # Forward and backward propagations
            time_forward, psi_traj = self.forward_propagation(efield)
            _, chi_traj = self.backward_propagation(efield, psi_traj)

            # Metrics
            final_psi = psi_traj[-1]
            fidelity = calculate_fidelity(final_psi, self.target_idx)
            field_data = efield.get_Efield()
            field_norm = float(np.linalg.norm(field_data))

            # Costs
            J_T = 1.0 - float(fidelity)
            E2 = field_data[:, 0] ** 2 + field_data[:, 1] ** 2
            J_a = 0.5 * float(LAMBDA_A) * float(np.sum(S_t * E2) * dt_field)
            J = J_T + J_a

            fidelity_history.append(fidelity)
            field_norm_history.append(field_norm)
            terminal_cost_history.append(J_T)
            running_cost_history.append(J_a)
            total_cost_history.append(J)

            if iteration % 10 == 0 or iteration < 10:
                print(
                    f"Iteration {iteration + 1:3d}: Fidelity = {fidelity:.6f}, "
                    f"Field norm = {field_norm:.2e}, J_T = {J_T:.6e}, J_a = {J_a:.6e}, J = {J:.6e}"
                )

            # Check target
            if fidelity >= target_fidelity:
                print(f"🎉 目標フィデリティー {target_fidelity} に到達しました！")
                break

            # Early stopping
            if iteration + 1 >= MIN_ITER_EARLYSTOP and fidelity >= EARLYSTOP_MIN_FIDELITY:
                recent = [abs(fidelity_history[-1] - fidelity_history[-k]) for k in range(1, min(3, len(fidelity_history)))]
                if len(recent) > 0 and all(r < convergence_tol for r in recent):
                    print(
                        f"フィデリティー変化が収束閾値 {convergence_tol} を下回り、"
                        f"Fidelity≥{EARLYSTOP_MIN_FIDELITY:.2f} のため終了します。"
                    )
                    break

            # Gradient and update (gradient descent on J)
            grad = self.compute_gradient_local(psi_traj, chi_traj, field_data, lambda_a, self.tlist)
            updated_field = field_data - float(learning_rate) * grad

            # Optional hard clipping (generally disabled for monotonic behavior)
            if ENFORCE_HARD_CLIP:
                updated_field[:, 0] = np.clip(updated_field[:, 0], -FIELD_MAX, FIELD_MAX)
                updated_field[:, 1] = np.clip(updated_field[:, 1], -FIELD_MAX, FIELD_MAX)

            # Re-apply envelope window (to preserve boundary conditions)
            if self.envelope_window is not None:
                updated_field = apply_envelope_to_field(self.tlist, updated_field, self.envelope_window)

            efield = ElectricField(tlist=self.tlist)
            efield.add_arbitrary_Efield(updated_field)

        elapsed_time = time.time() - start_time
        print(f"\n最適化完了 (所要時間: {elapsed_time:.2f}秒)")
        if len(fidelity_history) > 0:
            print(f"最終フィデリティー: {fidelity_history[-1]:.6f}")
            print(f"反復回数: {len(fidelity_history)}")

        # Save histories for plotting
        self.fidelity_history = fidelity_history
        self.field_norm_history = field_norm_history
        self.terminal_cost_history = terminal_cost_history
        self.running_cost_history = running_cost_history
        self.total_cost_history = total_cost_history

        return efield, fidelity_history, field_norm_history


# =========================
# Main and plotting
# =========================
def main() -> None:
    print("=== GRAPEによる振動回転励起最適制御 ===")
    print(f"基底設定: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")

    basis = LinMolBasis(
        V_max=V_MAX, J_max=J_MAX, use_M=USE_M,
        omega=OMEGA_01, delta_omega=DOMEGA, B=B_CONSTANT, alpha=ALPHA_CONSTANT,
        input_units=UNIT_FREQUENCY, output_units="rad/fs"
    )
    H0 = basis.generate_H0()

    dipole_matrix = LinMolDipoleMatrix(
        basis=basis,
        mu0=MU0,
        potential_type="harmonic",
        backend="numpy",
        dense=True,
        units="C*m",
        units_input=UNIT_DIPOLE,
    )

    print(f"基底次元: {basis.size()}")
    initial_idx = basis.get_index(INITIAL_STATE)
    target_idx = basis.get_index(TARGET_STATE)
    print(f"初期状態: {INITIAL_STATE} → インデックス {initial_idx}")
    print(f"目標状態: {TARGET_STATE} → インデックス {target_idx}")

    eigenvalues = H0.get_eigenvalues()
    initial_energy = eigenvalues[initial_idx]
    target_energy = eigenvalues[target_idx]
    energy_diff = target_energy - initial_energy
    print(f"初期状態エネルギー: {initial_energy:.6f} rad/fs")
    print(f"目標状態エネルギー: {target_energy:.6f} rad/fs")
    print(f"エネルギー差: {energy_diff:.6f} rad/fs")

    optimizer = GRAPEOptimizer(
        basis=basis,
        hamiltonian=H0,
        dipole_matrix=dipole_matrix,
        initial_idx=initial_idx,
        target_idx=target_idx,
        time_total=TIME_TOTAL,
        dt=DT_EFIELD,
        sample_stride=SAMPLE_STRIDE,
    )

    optimal_field, fidelity_hist, field_norm_hist = optimizer.optimize()

    plot_optimization_results(optimizer, optimal_field, fidelity_hist, field_norm_hist)


def plot_optimization_results(optimizer: GRAPEOptimizer, optimal_field: ElectricField,
                              fidelity_hist: list, field_norm_hist: list) -> None:
    # Forward propagation with the optimized field to get populations
    time_final, psi_final = optimizer.forward_propagation(optimal_field)
    prob_final = np.abs(psi_final) ** 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Optimized field
    field_data = optimal_field.get_Efield()
    axes[0, 0].plot(optimizer.tlist, field_data[:, 0], 'r-', label='Ex(t)', linewidth=1.5)
    axes[0, 0].plot(optimizer.tlist, field_data[:, 1], 'b-', label='Ey(t)', linewidth=1.5)
    axes[0, 0].set_xlabel('Time [fs]')
    axes[0, 0].set_ylabel('Electric Field [V/m]')
    axes[0, 0].set_title('Optimized Electric Field (GRAPE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Fidelity convergence
    iterations = range(1, len(fidelity_hist) + 1)
    axes[0, 1].plot(iterations, fidelity_hist, 'go-', linewidth=2, markersize=4)
    axes[0, 1].axhline(y=TARGET_FIDELITY, color='r', linestyle='--', alpha=0.7, label=f'Target: {TARGET_FIDELITY}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Fidelity')
    axes[0, 1].set_title('Convergence History')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # 3. Populations (highlight initial/target)
    highlight_states = [INITIAL_STATE, TARGET_STATE]
    for state in highlight_states:
        if state in optimizer.basis.index_map:
            idx = optimizer.basis.get_index(state)
            axes[1, 0].plot(time_final, prob_final[:, idx], linewidth=2.5,
                            label=f'|v={state[0]}, J={state[1]}, M={state[2]}⟩ (highlight)')

    for i, (v, J, M) in enumerate(optimizer.basis.basis):
        if (v, J, M) in highlight_states:
            continue
        axes[1, 0].plot(time_final, prob_final[:, i], linewidth=1.0, alpha=0.9,
                        label=f'|v={v}, J={J}, M={M}⟩')

    total_sum = np.sum(prob_final, axis=1)
    axes[1, 0].plot(time_final, total_sum, color='gray', linestyle='--', linewidth=2.0, label='Total sum')
    axes[1, 0].set_xlabel('Time [fs]')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_title('State Population Evolution')
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)

    # 4. Field norm and costs
    ax4 = axes[1, 1]
    ax4.plot(iterations, field_norm_hist, 'bo-', linewidth=2, markersize=4, label='||E||')
    if hasattr(optimizer, 'total_cost_history'):
        ax4.plot(iterations, optimizer.total_cost_history, 'k-', linewidth=2, label='J')
    if hasattr(optimizer, 'terminal_cost_history'):
        ax4.plot(iterations, optimizer.terminal_cost_history, 'g--', linewidth=1.5, label='J_T')
    if hasattr(optimizer, 'running_cost_history'):
        ax4.plot(iterations, optimizer.running_cost_history, 'r--', linewidth=1.5, label='J_a')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Metric')
    ax4.set_title('Field Norm and Cost Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Save figure under examples/figures [[memory:2714886]]
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"grape_optimization_results_{timestamp}.png"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"結果を保存しました: {filepath}")
    plt.show()


if __name__ == "__main__":
    main()


