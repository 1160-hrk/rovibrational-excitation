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
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# Add src and examples to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.propagation.utils import cm_to_rad_phz
from rovibrational_excitation.core.units.converters import converter
from utils.fft_utils import spectrogram_fast


# =========================
# Parameters
# =========================
# System parameters
V_MAX = 3
J_MAX = 2
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
TARGET_STATE = (3, 1, 0)

# Time grid
TIME_TOTAL = 300000.0  # [fs]
DT_EFIELD = 0.1      # [fs]
SAMPLE_STRIDE = 1

# Local optimization parameters
GAIN = 6.0e19          # 局所制御則のゲイン（振幅スケール）
FIELD_MAX = 1e12      # [V/m] 物理的クリップ
USE_SIN2_SHAPE = False  # 端点0化のための形状関数

# 使用する物理軸2つを指定（例: "xy", "zx", "yz"）。
# 左の軸がEx（出力x成分）、右の軸がEy（出力y成分）に対応します。
CONTROL_AXES = "zx"

# Seeding when initial overlap is zero (avoid E=0 fixed point)
SEED_AMPLITUDE = 1e3
SEED_MAX_SEGMENTS = 5  # シードを許可する最大区間数
C_ABS_MIN = 1e-1  # 目標状態との重みがこれ以下ならシード
SHAPE_FLOOR = 1e-2  # 形状関数の下限

# Lookahead (先読み) 設定: 区間の一部だけ自由進行してから勾配を評価
LOOKAHEAD_ENABLE = False
LOOKAHEAD_FRACTION = 0.7  # 先読み割合（0〜1）

# Evaluation operator settings
#   mode: "target" uses A=|phi><phi| (従来)
#         "weights" uses A=diag(w_i)（基底ごとの重み）
EVALUATION_MODE = "weights"  # "target" or "weights"
# EVALUATION_MODE = "target"  # "target" or "weights"
# WEIGHT_MODE options: "by_v", "by_v_power", "custom"
# 追加: 逆順もサポート → "by_v_reverse", "by_v_power_reverse"
WEIGHT_MODE = "by_v_power"         
WEIGHT_V_POWER = 0.25          # by_v_power 用の指数（例: 2 で v^2）
WEIGHT_TARGET_FACTOR = 1
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
SEGMENT_SIZE_FS: Optional[float] = 4

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "results" / f"{TIMESTAMP}_local_optimization_rovibrational"


# =========================
# Utilities
# =========================
def shape_function(t: np.ndarray, T: float) -> np.ndarray:
    return np.sin(np.pi * t / T) ** 2


def calculate_fidelity(psi_final: np.ndarray, target_idx: int) -> float:
    return float(np.abs(psi_final[target_idx]) ** 2)


def build_segments_and_tlist(time_total: float, dt: float,
                   seg_steps: Optional[int], seg_fs: Optional[float]) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    フィールド配列のインデックス上で [start, end) の半開区間のリストを返す。
    """
    if seg_steps is not None and seg_steps > 0:
        steps = int((seg_steps//2) * 2)
    elif seg_fs is not None and seg_fs > 0:
        steps = int((seg_fs / dt // 2) * 2)
    else:
        raise ValueError("SEGMENT_SIZE_STEPS または SEGMENT_SIZE_FS が指定されていません")
    segments_number = int(np.ceil(time_total / dt / steps))
    n_field_steps = segments_number * steps + 1
    tlist = np.arange(0.0, dt * (n_field_steps + 1), dt)
    segments: list[tuple[int, int]] = [
        (int(s*steps), int((s+1)*steps)) for s in range(segments_number)
    ]
    return segments, tlist


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
                 sample_stride: int = 1,
                 seg_steps: Optional[int] = SEGMENT_SIZE_STEPS,
                 seg_fs: Optional[float] = SEGMENT_SIZE_FS,
                 ) -> None:
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.dipole_matrix = dipole_matrix
        self.initial_idx = initial_idx
        self.target_idx = target_idx
        self.dt = float(dt)
        self.sample_stride = sample_stride
        self.segments, self.tlist = build_segments_and_tlist(time_total, dt, seg_steps, seg_fs)
        # RK4整合時間軸
        self.n_steps = len(self.tlist)
        self.n_traj_steps = self.n_steps // 2 // self.sample_stride

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
        mu_z_si = self.dipole_matrix.get_mu_z_SI()
        if hasattr(mu_x_si, 'toarray'):
            mu_x_si = mu_x_si.toarray()
        if hasattr(mu_y_si, 'toarray'):
            mu_y_si = mu_y_si.toarray()
        if hasattr(mu_z_si, 'toarray'):
            mu_z_si = mu_z_si.toarray()
        self.mu_x_prime = cm_to_rad_phz(mu_x_si)
        self.mu_y_prime = cm_to_rad_phz(mu_y_si)
        self.mu_z_prime = cm_to_rad_phz(mu_z_si)

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
            'z': self.mu_z_prime,
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
            # ターゲット状態の重みを同じ振動順位のものより WEIGHT_TARGET_FACTOR 倍だけ大きくする
            if one_hot_idx is None:
                try:
                    v_target = int(self.basis.basis[self.target_idx][0])
                    w_before = float(self.A_diag[self.target_idx])
                    self.A_diag[self.target_idx] = w_before * float(WEIGHT_TARGET_FACTOR)
                    print(f"  ターゲット重み強調: idx={self.target_idx}, v={v_target}, w={w_before:.3g} -> {float(self.A_diag[self.target_idx]):.3g} (x{WEIGHT_TARGET_FACTOR})")
                except Exception:
                    pass
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
        print(f"  区間数: {len(self.segments)}")
        print(f"  区間長: {self.segments[0][1] - self.segments[0][0]} 点 ({self.dt * (self.segments[0][1] - self.segments[0][0])} fs)")
        print(f"  初期状態: {basis.basis[initial_idx]} → idx={initial_idx}")
        print(f"  目標状態: {basis.basis[target_idx]} → idx={target_idx}")
        print(f"  物理軸の割当: {self.selected_axes[0]}→Ex, {self.selected_axes[1]}→Ey (伝播は 'xy')")
        print(f"  先読み: {'ON' if LOOKAHEAD_ENABLE else 'OFF'} (fraction={LOOKAHEAD_FRACTION})")

    def propagate_segment(self, psi0: np.ndarray, e_segment: np.ndarray, t_segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """単一区間の定数電場で伝播し、時間と軌跡を返す。"""
        ef = ElectricField(tlist=t_segment)
        ef.add_arbitrary_Efield(e_segment)
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

    def design_field(self,
                     gain: float = GAIN,
                     use_shape: bool = USE_SIN2_SHAPE,
                    ) -> Tuple[ElectricField, np.ndarray]:
        """区間ごとに定数電場を設計し、全体の ElectricField を返す。"""
        segments = self.segments
        full_field = np.zeros((self.n_steps, 2), dtype=float)

        psi_curr = self.psi_initial.copy()
        t_curr_idx = 0
        seed_left = SEED_MAX_SEGMENTS

        for (start, end) in segments:
            # 区間代表時刻（中央）
            mid = (start + end) // 2
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
                term_x = complex(np.vdot(psi_ref, A_mu_x_psi))
                term_y = complex(np.vdot(psi_ref, A_mu_y_psi))
                im_x = float(np.imag(term_x))
                im_y = float(np.imag(term_y))
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
                if abs(c) < C_ABS_MIN and seed_left > 0:
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
            full_field[start+1:end+1, 0] = ex
            full_field[start+1:end+1, 1] = ey

            # この区間の時間配列で実伝播（逐次）
            t_segment = self.tlist[start:end+1]
            _, psi_traj_seg = self.propagate_segment(psi_curr, full_field[start:end+1, :], t_segment)
            psi_curr = psi_traj_seg[-1]

        # 設計済みフィールドを構築
        ef_total = ElectricField(tlist=self.tlist)
        ef_total.add_arbitrary_Efield(full_field)
        # ランニングコスト J_a = 1/2 * λ * ∫ S(t) |E|^2 dt（表示用）
        field_penalty = 1/GAIN
        try:
            S_run = shape_function(self.tlist, self.tlist[-1]) if use_shape else np.ones_like(self.tlist)
            E2 = full_field[:, 0] ** 2 + full_field[:, 1] ** 2
            dt_field = float(self.tlist[1] - self.tlist[0])
            self.running_cost = float(field_penalty) * float(np.sum(S_run * E2) * dt_field)
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
def main() -> Tuple[LocalOptimizer, ElectricField, np.ndarray, np.ndarray]:
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

    return optimizer, efield, psi_traj, field_data

if __name__ == "__main__":
    optimizer, efield, psi_traj, field_data = main()

    # 共通の出力先と既定パラメータを先に初期化
    figures_dir = FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    fmin = float(max(0.0, OMEGA_01 - 500.0))
    fmax = float(OMEGA_01 + 500.0)
    Npad = 0

    # %% Figure: Designed Electric Field (Ex, Ey)
    try:
        t = optimizer.tlist
        plt.figure(figsize=(12, 4))
        plt.plot(t, field_data[:, 0], 'r-', label='Ex(t)')
        plt.plot(t, field_data[:, 1], 'b-', label='Ey(t)')
        plt.xlabel('Time [fs]')
        plt.ylabel('Electric Field [V/m]')
        plt.title('Designed Electric Field (Local Optimization)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(figures_dir, f"local_optimization_field_{TIMESTAMP}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
        plt.show()
    except Exception as e_plot_field:
        print(f"Electric Field プロットでエラー: {e_plot_field}")

    # %% Figure: Target Population vs Time
    try:
        prob = np.abs(psi_traj) ** 2
        t = optimizer.tlist
        idx_tar = optimizer.target_idx
        plt.figure(figsize=(12, 4))
        plt.plot(t[::2*SAMPLE_STRIDE], prob[:, idx_tar], 'g-')
        plt.xlabel('Time [fs]')
        plt.ylabel('Population of target')
        plt.title('Target Population vs Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        filepath = os.path.join(figures_dir, f"local_optimization_fidelity_{TIMESTAMP}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
        plt.show()
    except Exception as e_plot_fid:
        print(f"Fidelity プロットでエラー: {e_plot_fid}")

    # %% Figure: State Population Evolution
    try:
        t = optimizer.tlist
        prob = np.abs(psi_traj) ** 2
        plt.figure(figsize=(12, 6))
        def _norm_state(state: tuple[int, ...]) -> tuple[int, ...]:
            return normalize_state_for_basis(optimizer.basis, state)
        highlight_states = [_norm_state(INITIAL_STATE), _norm_state(TARGET_STATE)]
        highlight_set = {tuple(s) for s in highlight_states}
        for state in highlight_states:
            if state in optimizer.basis.index_map:
                idx = optimizer.basis.get_index(state)
                label = f'|v={state[0]}, J={state[1]}⟩ (highlight)' if len(state) == 2 else f'|v={state[0]}, J={state[1]}, M={state[2]}⟩ (highlight)'
                plt.plot(t[::2*SAMPLE_STRIDE], prob[:, idx], linewidth=2.5, label=label)
        for i, st in enumerate(optimizer.basis.basis):
            st_t = tuple(int(x) for x in st)
            if st_t in highlight_set:
                continue
            if len(st_t) == 2:
                v, J = st_t
                label = f'|v={v}, J={J}⟩'
            else:
                v, J, M = st_t
                label = f'|v={v}, J={J}, M={M}⟩'
            plt.plot(t[::2*SAMPLE_STRIDE], prob[:, i], linewidth=1.0, alpha=0.9, label=label)
        plt.xlabel('Time [fs]')
        plt.ylabel('Population')
        plt.title('State Population Evolution')
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        filepath = os.path.join(figures_dir, f"local_optimization_populations_{TIMESTAMP}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
        plt.show()
    except Exception as e_plot_pop:
        print(f"Population プロットでエラー: {e_plot_pop}")

    # %% Figure: Field Norm (overall)
    try:
        t = optimizer.tlist
        field_norm = np.linalg.norm(field_data)
        plt.figure(figsize=(8, 3))
        plt.plot([0, len(t)-1], [field_norm, field_norm], 'k-')
        plt.xlabel('Index')
        plt.ylabel('||E||')
        plt.title('Field Norm (overall)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(figures_dir, f"local_optimization_fieldnorm_{TIMESTAMP}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
        plt.show()
    except Exception as e_plot_norm:
        print(f"Field Norm プロットでエラー: {e_plot_norm}")

    # %% Figure: Spectrum (Ex/Ey intensity and phase) with transition lines
    # Npad は後段のスペクトログラム処理でも使用するため既定値で初期化
    Npad = 0
    try:
        t_fs = efield.get_time_SI()
        dt_fs = float(t_fs[1] - t_fs[0])
        E_t = efield.get_Efield()
        N = len(t_fs)
        df_target_PHz = float(converter.convert_frequency(0.1, "cm^-1", "PHz"))
        Npad = int(np.ceil(1.0 / (dt_fs * df_target_PHz)))
        Npad = max(Npad, N)
        E_freq = np.fft.rfft(E_t, n=Npad, axis=0)
        freq_PHz = np.fft.rfftfreq(Npad, d=dt_fs)
        freq_cm = np.asarray(converter.convert_frequency(freq_PHz, "PHz", "cm^-1"), dtype=float)
        t_center = optimizer.tlist[-1] / 2.0
        E_freq_comp = E_freq * (np.exp(1j * 2 * np.pi * freq_PHz * t_center)).reshape((len(freq_PHz), 1))
        intensity_x = np.abs(E_freq_comp[:, 0]) ** 2
        intensity_y = np.abs(E_freq_comp[:, 1]) ** 2
        intensity_sum = intensity_x + intensity_y
        peak_idx = int(np.argmax(intensity_sum))
        f0 = float(freq_cm[peak_idx])
        I0 = float(intensity_sum[peak_idx])
        I_th = I0 / np.exp(2.0)
        def _interp_cross(fa, Ia, fb, Ib, Ith):
            if Ib == Ia:
                return fb
            return float(fa + (Ith - Ia) * (fb - fa) / (Ib - Ia))
        f_left = float(freq_cm[0])
        found_left = False
        for i in range(peak_idx - 1, 0, -1):
            if intensity_sum[i] >= I_th and intensity_sum[i - 1] < I_th:
                f_left = _interp_cross(freq_cm[i], intensity_sum[i], freq_cm[i - 1], intensity_sum[i - 1], I_th)
                found_left = True
                break
        if not found_left:
            f_left = float(freq_cm[0])
        f_right = float(freq_cm[-1])
        found_right = False
        for i in range(peak_idx + 1, len(freq_cm)):
            if intensity_sum[i - 1] >= I_th and intensity_sum[i] < I_th:
                f_right = _interp_cross(freq_cm[i - 1], intensity_sum[i - 1], freq_cm[i], intensity_sum[i], I_th)
                found_right = True
                break
        if not found_right:
            f_right = float(freq_cm[-1])
        fmin = float(max(0.0, OMEGA_01 - 500.0))
        fmax = float(OMEGA_01 + 500.0)
        try:
            eigenvalues = optimizer.hamiltonian.get_eigenvalues()  # rad/fs
            states = optimizer.basis.basis
            energy_by_vj_map: dict[tuple[int, int], float] = {}
            for idx, st in enumerate(states):
                v = int(st[0]); J = int(st[1])
                M = int(st[2]) if len(st) == 3 else 0
                key = (v, J)
                if key not in energy_by_vj_map or M == 0:
                    energy_by_vj_map[key] = float(eigenvalues[idx])
            trans_wn: list[float] = []
            for (v, J), E0 in energy_by_vj_map.items():
                v_up = v + 1
                for dJ in (+1, -1):
                    key = (v_up, J + dJ)
                    if key in energy_by_vj_map:
                        d_omega = energy_by_vj_map[key] - E0
                        wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                        if np.isfinite(wn) and wn > 0:
                            trans_wn.append(wn)
            if len(trans_wn) >= 1:
                wn_min = float(np.min(trans_wn))
                wn_max = float(np.max(trans_wn))
                center = 0.5 * (wn_min + wn_max)
                span = max(wn_max - wn_min, 1e-6)
                factor = 2
                half = 0.5 * span * factor
                fmin = max(center - half, float(freq_cm[0]))
                fmax = min(center + half, float(freq_cm[-1]))
        except Exception:
            pass
        phase_x_raw = -np.unwrap(np.angle(E_freq_comp[:, 0]))
        phase_y_raw = -np.unwrap(np.angle(E_freq_comp[:, 1]))
        dphidk_x = np.gradient(phase_x_raw, freq_cm)
        dphidk_y = np.gradient(phase_y_raw, freq_cm)
        slope_x = float(dphidk_x[peak_idx])
        slope_y = float(dphidk_y[peak_idx])
        phase_x = phase_x_raw - (slope_x * (freq_cm - f0) + phase_x_raw[peak_idx])
        phase_y = phase_y_raw - (slope_y * (freq_cm - f0) + phase_y_raw[peak_idx])
        mask = (freq_cm >= fmin) & (freq_cm <= fmax)
        freq_p = freq_cm[mask]
        intensity_x_p = intensity_x[mask]
        intensity_y_p = intensity_y[mask]
        phase_x_p = phase_x[mask]
        phase_y_p = phase_y[mask]
        fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax0.plot(freq_p, intensity_x_p, color='tab:blue', label='|Ex|²')
        ax0.set_ylabel('Intensity (a.u.)')
        ax0_t = ax0.twinx()
        ax0_t.plot(freq_p, phase_x_p, color='tab:red', alpha=0.7, label='Phase Ex')
        ax0_t.set_ylabel('Phase (rad)')
        ax0.set_title('Designed Field Spectrum (Ex)')
        ax0.set_xlim(fmin, fmax)
        lines0, labels0 = ax0.get_legend_handles_labels()
        lines0_t, labels0_t = ax0_t.get_legend_handles_labels()
        ax0.legend(lines0 + lines0_t, labels0 + labels0_t, loc='upper right')
        ax1.plot(freq_p, intensity_y_p, color='tab:green', label='|Ey|²')
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1_t = ax1.twinx()
        ax1_t.plot(freq_p, phase_y_p, color='tab:orange', alpha=0.7, label='Phase Ey')
        ax1_t.set_ylabel('Phase (rad)')
        ax1.set_title('Designed Field Spectrum (Ey)')
        ax1.set_xlim(fmin, fmax)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_t, labels1_t = ax1_t.get_legend_handles_labels()
        ax1.legend(lines1 + lines1_t, labels1 + labels1_t, loc='upper right')
        try:
            eigenvalues = optimizer.hamiltonian.get_eigenvalues()
            states = optimizer.basis.basis
            energy_by_vj_lines: dict[tuple[int, int], float] = {}
            for idx, st in enumerate(states):
                v = int(st[0]); J = int(st[1])
                M = int(st[2]) if len(st) == 3 else 0
                key = (v, J)
                if key not in energy_by_vj_lines or M == 0:
                    energy_by_vj_lines[key] = float(eigenvalues[idx])
            lines_vj: list[tuple[float, str]] = []
            for (v, J), E0 in energy_by_vj_lines.items():
                v_up = v + 1
                key_R = (v_up, J + 1)
                if key_R in energy_by_vj_lines:
                    d_omega = energy_by_vj_lines[key_R] - E0
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        lines_vj.append((wn, rf"$R({J})_{{{v}}}$"))
                key_P = (v_up, J - 1)
                if key_P in energy_by_vj_lines:
                    d_omega = energy_by_vj_lines[key_P] - E0
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        lines_vj.append((wn, rf"$P({J})_{{{v}}}$"))
            y0 = float(np.max(intensity_x_p)) if intensity_x_p.size else 1.0
            y1 = float(np.max(intensity_y_p)) if intensity_y_p.size else 1.0
            for wn, lbl in lines_vj:
                if fmin <= wn <= fmax:
                    ax0.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                    ax0.text(wn, y0, lbl, rotation=-90, va='top', ha='center', fontsize=8, color='gray')
                    ax1.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                    ax1.text(wn, y1, lbl, rotation=-90, va='top', ha='center', fontsize=8, color='gray')
        except Exception as e_lines:
            print(f"遷移線オーバーレイでエラー: {e_lines}")
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath2 = os.path.join(figures_dir, f"local_optimization_spectrum_{timestamp}.png")
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"スペクトル図を保存しました: {filepath2}")
        plt.show()
    except Exception as e_spec:
        print(f"スペクトル可視化でエラー: {e_spec}")
        if 'fmin' not in locals():
            fmin = float(max(0.0, OMEGA_01 - 500.0))
        if 'fmax' not in locals():
            fmax = float(OMEGA_01 + 500.0)
        if 'Npad' not in locals():
            Npad = 0

    # %% Figure: Spectrogram (Ex) with peak ridge and transition hlines
    t_fs = efield.get_time_SI()
    Ex = efield.get_Efield()[:, 0]
    if 'fmin' not in locals() or 'fmax' not in locals():
        center = float(OMEGA_01)
        span = 500.0
        fmin = center - span
        fmax = center + span
    T_index = len(t_fs) // 20
    res = spectrogram_fast(t_fs, Ex, T=T_index, unit_T='index', window_type='hamming', step=max(1, T_index // 8), N_pad=Npad)
    if len(res) == 4:
        x_spec, freq_1fs, spec, _max_idx = res
    else:
        x_spec, freq_1fs, spec = res
    freq_cm_full = np.asarray(converter.convert_frequency(freq_1fs, "PHz", "cm^-1"), dtype=float)
    mask_rng = (freq_cm_full >= fmin) & (freq_cm_full <= fmax)
    freq_cm_plot = freq_cm_full[mask_rng]
    spec_plot = spec[mask_rng, :]
    X, Y = np.meshgrid(x_spec, freq_cm_plot)
    # %% Figure: Spectrogram (Ex) with peak ridge and transition hlines
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    cf = ax3.pcolormesh(X, Y, spec_plot, shading='auto', cmap='viridis')
    ax3.set_xlabel('Time [fs]')
    ax3.set_ylabel('Wavenumber (cm$^{-1}$)')
    ax3.set_title('Spectrogram (Ex)')
    ax3.set_ylim(fmin, fmax)
    fig3.colorbar(cf, ax=ax3, label='|FFT|')
    try:
        if spec_plot.ndim == 2 and spec_plot.size > 0:
            peak_indices = np.argmax(spec_plot, axis=0)
            ridge_cm = freq_cm_plot[peak_indices]
            ax3.plot(x_spec, ridge_cm, color='red', linewidth=1.2, label='Peak frequency', zorder=4)
    except Exception as e_ridge:
        print(f"ピークリッジ描画でエラー: {e_ridge}")
    try:
        eigenvalues = optimizer.hamiltonian.get_eigenvalues()  # rad/fs
        states = optimizer.basis.basis
        energy_by_vj_lines_spec: dict[tuple[int, int], float] = {}
        for idx, st in enumerate(states):
            v = int(st[0]); J = int(st[1])
            M = int(st[2]) if len(st) == 3 else 0
            key = (v, J)
            if key not in energy_by_vj_lines_spec or M == 0:
                energy_by_vj_lines_spec[key] = float(eigenvalues[idx])
        lines_wn_spec: list[float] = []
        for (v, J), E0 in energy_by_vj_lines_spec.items():
            v_up = v + 1
            for dJ in (+1, -1):
                key = (v_up, J + dJ)
                if key in energy_by_vj_lines_spec:
                    d_omega = energy_by_vj_lines_spec[key] - E0
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        lines_wn_spec.append(wn)
        for wn in lines_wn_spec:
            if fmin <= wn <= fmax:
                ax3.hlines(wn, x_spec[0], x_spec[-1], colors='white', linestyles='-', linewidth=0.6, alpha=0.4, zorder=3)
        handles, labels = ax3.get_legend_handles_labels()
        if handles:
            ax3.legend(loc='upper right')
    except Exception as e_hlines:
        print(f"スペクトログラムの遷移線でエラー: {e_hlines}")
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath3 = os.path.join(figures_dir, f"local_optimization_spectrogram_{timestamp}.png")
    plt.savefig(filepath3, dpi=300, bbox_inches='tight')
    print(f"スペクトログラム図を保存しました: {filepath3}")
    plt.show()


