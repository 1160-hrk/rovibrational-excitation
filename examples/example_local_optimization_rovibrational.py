#!/usr/bin/env python
"""
局所最適化理論（Local Optimization / Local Control）による
振動回転励起の電場設計（区間サイズ選択対応）
=====================================================

本スクリプトは、目標射影演算子 A = |phi><phi| の増加を局所的（時間局所）に
最大化する制御則に基づき、区間ごとに定数電場を決定して逐次伝播します。

区間サイズは、(a) フィールド配列ステップ数 もしくは (b) 時間[fs] の
いずれかで指定可能です。

実行:
  python examples/local_optimization_rovibrational.py
"""

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.propagation.utils import cm_to_rad_phz


# =========================
# Parameters
# =========================
# System parameters
V_MAX = 1
J_MAX = 1
USE_M = False

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
TARGET_STATE = (1, 1, 0)

# Time grid
TIME_TOTAL = 500.0  # [fs]
DT_EFIELD = 0.02      # [fs]
SAMPLE_STRIDE = 1

# Local optimization parameters
GAIN = 1e22          # 局所制御則のゲイン（振幅スケール）
FIELD_MAX = 1e12      # [V/m] 物理的クリップ
USE_SIN2_SHAPE = False  # 端点0化のための形状関数

# 使用する物理軸2つを指定（例: "xy", "zx", "yz"）。
# 左の軸がEx（出力x成分）、右の軸がEy（出力y成分）に対応します。
CONTROL_AXES = "zx"

# Seeding when initial overlap is zero (avoid E=0 fixed point)
SEED_ENABLE = True
SEED_AMPLITUDE = 1e8
SEED_MAX_SEGMENTS = 20
C_ABS_MIN = 1e-1
SHAPE_FLOOR = 1e-2

# Lookahead (先読み) 設定: 区間の一部だけ自由進行してから勾配を評価
LOOKAHEAD_ENABLE = True
LOOKAHEAD_FRACTION = 0.7  # 先読み割合（0〜1）

# Evaluation operator settings
#   mode: "target" uses A=|phi><phi| (従来)
#         "weights" uses A=diag(w_i)（基底ごとの重み）
EVALUATION_MODE = "weights"  # "target" or "weights"
# EVALUATION_MODE = "target"  # "target" or "weights"
# WEIGHT_MODE options: "by_v", "by_v_power", "custom"
# 追加: 逆順もサポート → "by_v_reverse", "by_v_power_reverse"
WEIGHT_MODE = "by_v_power"         
WEIGHT_V_POWER = 4.0          # by_v_power 用の指数（例: 2 で v^2）
NORMALIZE_WEIGHTS = True
# 逆順を強制するフラグ（WEIGHT_MODE が *_reverse の場合も有効）
WEIGHT_REVERSE = False
# カスタム重みの指定（どちらかを利用）
# 1) 基底次元に一致する長さの配列（list / np.ndarray）
CUSTOM_WEIGHTS = None
# 2) 状態タプル (v,J,M) をキーにした辞書（未指定キーは0）
CUSTOM_WEIGHTS_DICT = None
# 便利機能: weightsモードでターゲットのみ1（他0）にしたい場合
USE_ONE_HOT_TARGET_IN_WEIGHTS = False
# weightsモードで駆動量が極小のときのシード閾値
DRIVE_ABS_MIN = 1e-18

# Segmenting (either steps or duration). If both set, STEPS has priority.
SEGMENT_SIZE_STEPS: Optional[int] = None
SEGMENT_SIZE_FS: Optional[float] = 0.1


# =========================
# Utilities
# =========================
def shape_function(t: np.ndarray, T: float) -> np.ndarray:
    return np.sin(np.pi * t / T) ** 2


def calculate_fidelity(psi_final: np.ndarray, target_idx: int) -> float:
    return float(np.abs(psi_final[target_idx]) ** 2)


def build_segments(n_field_steps: int, tlist: np.ndarray,
                   seg_steps: Optional[int], seg_fs: Optional[float]) -> list[tuple[int, int]]:
    """
    フィールド配列のインデックス上で [start, end) の半開区間のリストを返す。
    """
    if seg_steps is not None and seg_steps > 0:
        steps = int(seg_steps)
    elif seg_fs is not None and seg_fs > 0:
        dt_field = float(tlist[1] - tlist[0])
        steps = max(1, int(round(seg_fs / dt_field)))
    else:
        steps = max(1, n_field_steps // 50)  # デフォルト: おおよそ50分割

    segments: list[tuple[int, int]] = []
    start = 0
    while start < n_field_steps - 1:
        end = min(n_field_steps, start + steps)
        if end <= start:
            end = start + 1
        segments.append((start, end))
        start = end
    return segments


def normalize_state_for_basis(basis: LinMolBasis, state: tuple[int, ...]) -> tuple[int, ...]:
    """use_M の有無に応じて状態タプルの次元を整える。
    - basis.use_M==False かつ (v,J,M) が来たら (v,J) に縮約
    - basis.use_M==True かつ (v,J) が来たら M=0 を付与
    それ以外はそのまま返す
    """
    use_m = getattr(basis, "use_M", USE_M)
    if use_m:
        if len(state) == 2:
            return (int(state[0]), int(state[1]), 0)
        return tuple(int(x) for x in state)
    # use_m == False
    if len(state) == 3:
        return (int(state[0]), int(state[1]))
    return tuple(int(x) for x in state)


def build_weights_for_basis(basis: LinMolBasis, *, mode: str = "by_v",
                            normalize: bool = True,
                            custom: Optional[np.ndarray] = None,
                            custom_dict: Optional[dict] = None,
                            v_power: float = 2.0,
                            one_hot_target_idx: Optional[int] = None,
                            reverse: bool = False) -> np.ndarray:
    """基底ごとの重みベクトル w_i を生成。
    - mode=="by_v": 各基底 (v,J,M) に対し w_i = v
    - mode=="by_v_power": w_i = v**v_power
    - mode=="custom": `custom` または `custom_dict` を使用
    - one_hot_target_idx が与えられた場合は、そのインデックスのみ1、他0
    正規化: max(w)>0 のとき w/=max(w)
    """
    dim = basis.size()
    if one_hot_target_idx is not None:
        w = np.zeros(dim, dtype=float)
        if 0 <= int(one_hot_target_idx) < dim:
            w[int(one_hot_target_idx)] = 1.0
    elif mode == "custom":
        if custom is not None:
            w = np.asarray(custom, dtype=float)
            if w.shape[0] != dim:
                raise ValueError(f"CUSTOM_WEIGHTS の長さ {w.shape[0]} が基底次元 {dim} と一致しません")
        elif custom_dict is not None:
            w = np.zeros(dim, dtype=float)
            for i, key in enumerate(basis.basis):
                w[i] = float(custom_dict.get(tuple(key), 0.0))
        else:
            raise ValueError("CUSTOM_WEIGHTS / CUSTOM_WEIGHTS_DICT のいずれも指定されていません")
    else:
        w = np.zeros(dim, dtype=float)
        for i, state in enumerate(basis.basis):
            v = int(state[0])  # USE_M=False でも先頭は v
            if mode == "by_v_power":
                w[i] = float(v) ** float(v_power)
            else:
                # by_v
                w[i] = float(v)
    if normalize:
        wmax = float(np.max(w))
        if wmax > 0:
            w = w / wmax
    # 逆順適用（one-hot時は無効）
    if one_hot_target_idx is None and reverse:
        if normalize:
            # [0,1] で反転
            w = 1.0 - w
        else:
            wmax = float(np.max(w))
            w = (wmax - w)
    return w


class LocalOptimizer:
    """局所最適化（Local Control）に基づく逐次設計。

    区間ごとに電場を一定とし、区間開始時点の状態 |psi> を用いて
    E_x, E_y を以下で決定する:
        E_ax = GAIN * S(t_c) * Im( <phi|psi>^* * <phi|mu_a'|psi> ), a in {x, y}
    ここで mu' は伝播ユニットに整合済み（cm_to_rad_phz を使用）
    """

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

        # RK4整合時間軸
        target_traj_steps = int(time_total / dt) + 1
        required_field_steps = 2 * (target_traj_steps - 1) + 1
        self.tlist = np.linspace(0.0, time_total, required_field_steps)
        self.n_steps = len(self.tlist)
        self.n_traj_steps = target_traj_steps

        self.propagator = SchrodingerPropagator(
            backend="numpy", validate_units=True, renorm=True
        )

        self.psi_initial = np.zeros(basis.size(), dtype=complex)
        self.psi_initial[initial_idx] = 1.0
        self.psi_target = np.zeros(basis.size(), dtype=complex)
        self.psi_target[target_idx] = 1.0

        # μ'（伝播単位系）
        mu_x_si = self.dipole_matrix.get_mu_x_SI()
        mu_y_si = self.dipole_matrix.get_mu_y_SI()
        if hasattr(mu_x_si, 'toarray'):
            mu_x_si = mu_x_si.toarray()
        if hasattr(mu_y_si, 'toarray'):
            mu_y_si = mu_y_si.toarray()
        self.mu_x_prime = cm_to_rad_phz(mu_x_si)
        self.mu_y_prime = cm_to_rad_phz(mu_y_si)
        # z成分
        if hasattr(self.dipole_matrix, 'get_mu_z_SI'):
            mu_z_si = self.dipole_matrix.get_mu_z_SI()
            if hasattr(mu_z_si, 'toarray'):
                mu_z_si = mu_z_si.toarray()
            self.mu_z_prime = cm_to_rad_phz(mu_z_si)
        else:
            self.mu_z_prime = None

        # 物理軸→出力Ex/Eyのマッピング
        sel = str(CONTROL_AXES).lower()
        if len(sel) != 2 or any(c not in 'xyz' for c in sel):
            sel = 'xy'
        self.selected_axes = sel
        # 伝播は常に2成分（x,y）で渡す
        self.propagation_axes = CONTROL_AXES
        mu_map = {
            'x': self.mu_x_prime,
            'y': self.mu_y_prime,
            'z': self.mu_z_prime if self.mu_z_prime is not None else self.mu_x_prime,
        }
        self.mu_eff_x_prime = mu_map[sel[0]]
        self.mu_eff_y_prime = mu_map[sel[1]]

        # 自由進行用の固有値（rad/fs）をキャッシュ
        try:
            self._eigenvalues = self.hamiltonian.get_eigenvalues()
            print(f"  固有値: {self._eigenvalues}")
        except Exception:
            self._eigenvalues = None

        # 評価演算子のセットアップ
        self.eval_mode = str(EVALUATION_MODE).lower()
        if self.eval_mode == "weights":
            one_hot_idx = self.target_idx if bool(USE_ONE_HOT_TARGET_IN_WEIGHTS) else None
            mode_str = str(WEIGHT_MODE).lower()
            mode_is_reverse = mode_str.endswith("_reverse")
            base_mode = mode_str.replace("_reverse", "")
            reverse_flag = bool(WEIGHT_REVERSE) or mode_is_reverse
            self.A_diag = build_weights_for_basis(
                basis,
                mode=base_mode,
                normalize=bool(NORMALIZE_WEIGHTS),
                custom=np.asarray(CUSTOM_WEIGHTS, dtype=float) if CUSTOM_WEIGHTS is not None else None,
                custom_dict=dict(CUSTOM_WEIGHTS_DICT) if CUSTOM_WEIGHTS_DICT is not None else None,
                v_power=float(WEIGHT_V_POWER),
                one_hot_target_idx=one_hot_idx,
                reverse=reverse_flag,
            )
            print("評価モード: weights (対角重み)")
            print(f"  重み min/max = {float(np.min(self.A_diag)):.3g}/{float(np.max(self.A_diag)):.3g}")
            if reverse_flag and one_hot_idx is None:
                print("  注記: 逆順重み（reverse）を適用しています")
            # 重みの要約を表示（上位/下位を少数表示）
            try:
                idx_sorted = np.argsort(self.A_diag)
                show = min(5, len(idx_sorted))
                print("  上位重み: ")
                for k in idx_sorted[::-1][:show]:
                    print(f"    idx={k}, state={basis.basis[k]}, w={self.A_diag[k]:.3g}")
                print("  下位重み: ")
                for k in idx_sorted[:show]:
                    print(f"    idx={k}, state={basis.basis[k]}, w={self.A_diag[k]:.3g}")
            except Exception:
                pass
        else:
            # target projector mode uses A = |phi><phi|（内部で専用式を使用）
            self.A_diag = None
            print("評価モード: target (|phi><phi|)")

        print("初期化完了:")
        print(f"  基底次元: {basis.size()}")
        print(f"  電場時間ステップ数: {self.n_steps}")
        print(f"  軌跡配列長(予想): {self.n_traj_steps}")
        print(f"  初期状態: {basis.basis[initial_idx]} → idx={initial_idx}")
        print(f"  目標状態: {basis.basis[target_idx]} → idx={target_idx}")
        print(f"  物理軸の割当: {self.selected_axes[0]}→Ex, {self.selected_axes[1]}→Ey (伝播は 'xy')")
        print(f"  先読み: {'ON' if LOOKAHEAD_ENABLE else 'OFF'} (fraction={LOOKAHEAD_FRACTION})")

    def propagate_segment(self, psi0: np.ndarray, e_segment: np.ndarray, t_segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """単一区間の定数電場で伝播し、時間と軌跡を返す。"""
        ef = ElectricField(tlist=t_segment)
        ef.add_arbitrary_Efield(np.tile(e_segment, (len(t_segment), 1)))
        result = self.propagator.propagate(
            hamiltonian=self.hamiltonian,
            efield=ef,
            dipole_matrix=self.dipole_matrix,
            initial_state=psi0,
            axes=self.propagation_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=self.sample_stride,
            algorithm="rk4",
            sparse=True,
        )
        return result[0], result[1]

    def design_field(self, seg_steps: Optional[int] = SEGMENT_SIZE_STEPS,
                     seg_fs: Optional[float] = SEGMENT_SIZE_FS,
                     gain: float = GAIN,
                     use_shape: bool = USE_SIN2_SHAPE,
                     field_penalty: float = FIELD_PENALTY) -> Tuple[ElectricField, np.ndarray]:
        """区間ごとに定数電場を設計し、全体の ElectricField を返す。"""
        segments = build_segments(self.n_steps, self.tlist, seg_steps, seg_fs)
        full_field = np.zeros((self.n_steps, 2), dtype=float)

        psi_curr = self.psi_initial.copy()
        t_curr_idx = 0
        seed_left = SEED_MAX_SEGMENTS if SEED_ENABLE else 0

        for (start, end) in segments:
            # 区間代表時刻（中央）
            mid = (start + end - 1) // 2
            S = shape_function(self.tlist, self.tlist[-1])[mid] if use_shape else 1.0
            S_eff = max(S, SHAPE_FLOOR) if use_shape else 1.0

            # 先読み: 半区間（比率）だけ H0 で自由進行した状態で勾配を評価
            psi_ref = psi_curr
            if LOOKAHEAD_ENABLE and LOOKAHEAD_FRACTION > 0:
                if self._eigenvalues is not None and end - start >= 1:
                    tau = float(self.tlist[max(start, 0)] - self.tlist[0])  # dummy init
                    tau = LOOKAHEAD_FRACTION * float(self.tlist[end-1] - self.tlist[start])
                    if tau > 0:
                        phase = np.exp(-1j * self._eigenvalues * tau)
                        psi_ref = psi_curr * phase

            # 局所制御則（モード切替）
            if self.eval_mode == "weights":
                # A = diag(w) のとき、E_a ∝ Im⟨ψ|[A, μ_a']|ψ⟩
                mu_x = self.mu_eff_x_prime
                mu_y = self.mu_eff_y_prime
                A_diag = self.A_diag
                # term1 = <ψ|A μ ψ>, term2 = <ψ| μ A ψ>
                mu_x_psi = mu_x @ psi_ref
                mu_y_psi = mu_y @ psi_ref
                A_mu_x_psi = A_diag * mu_x_psi
                A_mu_y_psi = A_diag * mu_y_psi
                A_psi = A_diag * psi_ref
                mu_x_A_psi = mu_x @ A_psi
                mu_y_A_psi = mu_y @ A_psi
                term1_x = complex(np.vdot(psi_ref, A_mu_x_psi))
                term2_x = complex(np.vdot(psi_ref, mu_x_A_psi))
                comm_x = term1_x - term2_x
                term1_y = complex(np.vdot(psi_ref, A_mu_y_psi))
                term2_y = complex(np.vdot(psi_ref, mu_y_A_psi))
                comm_y = term1_y - term2_y
                im_x = float(np.imag(comm_x))
                im_y = float(np.imag(comm_y))
                ex = float(gain * S * im_x)
                ey = float(gain * S * im_y)
                # 駆動が極小のときシード
                if (abs(im_x) < DRIVE_ABS_MIN and abs(im_y) < DRIVE_ABS_MIN and seed_left > 0):
                    ex = SEED_AMPLITUDE * S_eff
                    ey = SEED_AMPLITUDE * S_eff
                    seed_left -= 1
                    print(f"[seed] 区間[{start}:{end}] weights駆動が極小のためシード: Ex={ex:.3e}, Ey={ey:.3e}")
            else:
                # target projector モード
                c = complex(np.vdot(self.psi_target, psi_ref))
                d_x = complex(np.vdot(self.psi_target, (self.mu_eff_x_prime @ psi_ref)))
                d_y = complex(np.vdot(self.psi_target, (self.mu_eff_y_prime @ psi_ref)))
                val_x = float(np.imag(np.conj(c) * d_x))
                val_y = float(np.imag(np.conj(c) * d_y))
                ex = float(gain * S * val_x)
                ey = float(gain * S * val_y)
                if abs(c) < C_ABS_MIN and seed_left > 0 and SEED_ENABLE:
                    sx = 1.0 if (abs(d_x) == 0.0) else (1.0 if (np.real(d_x) >= 0.0) else -1.0)
                    sy = 1.0 if (abs(d_y) == 0.0) else (1.0 if (np.real(d_y) >= 0.0) else -1.0)
                    ex = sx * SEED_AMPLITUDE * S_eff
                    ey = sy * SEED_AMPLITUDE * S_eff
                    seed_left -= 1
                    print(f"[seed] 区間[{start}:{end}] にシード電場を適用: Ex={ex:.3e}, Ey={ey:.3e}, |c|={abs(c):.3e}")

            # 振幅制限
            ex = float(np.clip(ex, -FIELD_MAX, FIELD_MAX))
            ey = float(np.clip(ey, -FIELD_MAX, FIELD_MAX))

            # 区間フィールドをセット
            full_field[start:end, 0] = ex
            full_field[start:end, 1] = ey

            # この区間の時間配列で実伝播（逐次）
            t_segment = self.tlist[start:end]
            # ElectricFieldは少なくとも2点必要（dt計算のため）
            if len(t_segment) < 2:
                if end < self.n_steps:
                    t_propag = self.tlist[start:end + 1]
                elif start > 0:
                    t_propag = self.tlist[start - 1:end]
                else:
                    # フォールバック（理論上ここには来ない想定）
                    t_propag = self.tlist[0:2]
            else:
                t_propag = t_segment
            _, psi_traj_seg = self.propagate_segment(psi_curr, np.array([ex, ey], dtype=float), t_propag)
            psi_curr = psi_traj_seg[-1]
            t_curr_idx = end

        # 設計済みフィールドを構築
        ef_total = ElectricField(tlist=self.tlist)
        ef_total.add_arbitrary_Efield(full_field)
        # ランニングコスト J_a = 1/2 * λ * ∫ S(t) |E|^2 dt（表示用）
        try:
            if USE_PENALTY_SCALING and field_penalty is not None and field_penalty > 0:
                S_run = shape_function(self.tlist, self.tlist[-1]) if use_shape else np.ones_like(self.tlist)
                E2 = full_field[:, 0] ** 2 + full_field[:, 1] ** 2
                dt_field = float(self.tlist[1] - self.tlist[0])
                self.running_cost = 0.5 * float(field_penalty) * float(np.sum(S_run * E2) * dt_field)
            else:
                self.running_cost = None
        except Exception:
            self.running_cost = None
        return ef_total, full_field

    def forward_full(self, efield: ElectricField) -> Tuple[np.ndarray, np.ndarray]:
        result = self.propagator.propagate(
            hamiltonian=self.hamiltonian,
            efield=efield,
            dipole_matrix=self.dipole_matrix,
            initial_state=self.psi_initial,
            axes=self.propagation_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=self.sample_stride,
            algorithm="rk4",
            sparse=True,
        )
        return result[0], result[1]


# =========================
# Main & Plotting
# =========================
def main() -> None:
    print("=== 局所最適化理論による振動回転励起 電場設計 ===")
    print(f"基底設定: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")

    basis = LinMolBasis(
        V_max=V_MAX, J_max=J_MAX, use_M=USE_M,
        omega=OMEGA_01, delta_omega=DOMEGA, B=B_CONSTANT, alpha=ALPHA_CONSTANT,
        input_units=UNIT_FREQUENCY, output_units="rad/fs",
    )
    H0 = basis.generate_H0()

    dipole_matrix = LinMolDipoleMatrix(
        basis=basis,
        mu0=MU0,
        potential_type="harmonic",
        backend="numpy",
        dense=False,
        units="C*m",
        units_input=UNIT_DIPOLE,
    )

    init_state = normalize_state_for_basis(basis, INITIAL_STATE)
    targ_state = normalize_state_for_basis(basis, TARGET_STATE)
    initial_idx = basis.get_index(init_state)
    target_idx = basis.get_index(targ_state)
    print(f"初期状態: {init_state} (idx={initial_idx})")
    print(f"目標状態: {targ_state} (idx={target_idx})")

    optimizer = LocalOptimizer(
        basis=basis,
        hamiltonian=H0,
        dipole_matrix=dipole_matrix,
        initial_idx=initial_idx,
        target_idx=target_idx,
        time_total=TIME_TOTAL,
        dt=DT_EFIELD,
        sample_stride=SAMPLE_STRIDE,
    )

    start = time.time()
    efield, field_data = optimizer.design_field()
    elapsed = time.time() - start
    print(f"設計完了: {elapsed:.2f} s")
    if hasattr(optimizer, 'running_cost') and optimizer.running_cost is not None:
        print(f"ランニングコスト J_a (λ/2 ∫ S|E|^2 dt): {optimizer.running_cost:.6e}")

    # 最終評価（全区間をまとめて前進）
    time_full, psi_traj = optimizer.forward_full(efield)
    fidelity = calculate_fidelity(psi_traj[-1], optimizer.target_idx)
    print(f"最終フィデリティー: {fidelity:.6f}")

    plot_results(optimizer, efield, psi_traj, field_data)


def plot_results(optimizer: LocalOptimizer, efield: ElectricField, psi_traj: np.ndarray,
                 field_data: np.ndarray) -> None:
    prob = np.abs(psi_traj) ** 2
    t = optimizer.tlist

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Field
    axes[0, 0].plot(t, field_data[:, 0], 'r-', label='Ex(t)')
    axes[0, 0].plot(t, field_data[:, 1], 'b-', label='Ey(t)')
    axes[0, 0].set_xlabel('Time [fs]')
    axes[0, 0].set_ylabel('Electric Field [V/m]')
    axes[0, 0].set_title('Designed Electric Field (Local Optimization)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Fidelity (instantaneous target population)
    idx_tar = optimizer.target_idx
    axes[0, 1].plot(t[::2*SAMPLE_STRIDE], prob[:, idx_tar], 'g-')
    axes[0, 1].set_xlabel('Time [fs]')
    axes[0, 1].set_ylabel('Population of target')
    axes[0, 1].set_title('Target Population vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # 3) Populations (highlight initial/target)
    def _norm_state(state: tuple[int, ...]) -> tuple[int, ...]:
        return normalize_state_for_basis(optimizer.basis, state)

    highlight_states = [_norm_state(INITIAL_STATE), _norm_state(TARGET_STATE)]
    highlight_set = {tuple(s) for s in highlight_states}
    for state in highlight_states:
        if state in optimizer.basis.index_map:
            idx = optimizer.basis.get_index(state)
            label = f'|v={state[0]}, J={state[1]}⟩ (highlight)' if len(state) == 2 else f'|v={state[0]}, J={state[1]}, M={state[2]}⟩ (highlight)'
            axes[1, 0].plot(t[::2*SAMPLE_STRIDE], prob[:, idx], linewidth=2.5, label=label)
    for i, st in enumerate(optimizer.basis.basis):
        # st は (v,J) または (v,J,M)
        st_t = tuple(int(x) for x in st)
        if st_t in highlight_set:
            continue
        if len(st_t) == 2:
            v, J = st_t
            label = f'|v={v}, J={J}⟩'
        else:
            v, J, M = st_t
            label = f'|v={v}, J={J}, M={M}⟩'
        axes[1, 0].plot(t[::2*SAMPLE_STRIDE], prob[:, i], linewidth=1.0, alpha=0.9, label=label)
    axes[1, 0].set_xlabel('Time [fs]')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_title('State Population Evolution')
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)

    # 4) Field norm (per segment view is implicit); show overall norm
    field_norm = np.linalg.norm(field_data)
    axes[1, 1].plot([0, len(t)-1], [field_norm, field_norm], 'k-')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('||E||')
    axes[1, 1].set_title('Field Norm (overall)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure under examples/figures [[memory:2714886]]
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"local_optimization_results_{timestamp}.png"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"結果を保存しました: {filepath}")
    plt.show()


if __name__ == "__main__":
    main()


