#!/usr/bin/env python
"""
局所最適化理論（Local Optimization / Local Control）による
振動励起の電場設計
=====================================================

本スクリプトは、指定した評価関数を局所的に
最大化する制御則に基づき、区間ごとに定数電場を決定して逐次的に時間発展します。

区間サイズは、(a) 時間配列ステップ数 もしくは (b) 時間[fs] の
いずれかで指定可能です。

評価関数は、以下のいずれかを指定できます。
- "target": ターゲットとの内積
- "weights": 準位ごとの重み

重みは、以下のいずれかを指定できます。
- "by_v_power": 準位ごとの重み = 準位^v_power
- "custom": カスタム重み

カスタム重みは、以下のいずれかを指定できます。
- 基底次元に一致する長さの配列（list / np.ndarray）
- 状態タプル (v,) をキーにした辞書（未指定キーは0）

よく変更するパラメータ一覧
- GAIN: 局所制御則のゲイン（1e19-1e21程度が適切）
- V_MAX: 最大振動準位
- WN_01: ０→１準位への遷移周波数
- DELTA_WN: 遷移周波数の非調和シフト
- MU01: ０→１準位への双極子モーメント
- TIME_TOTAL: 総時間 (GAINが小さい場合は評価関数が最大化されずに終了する。)
- DT_EFIELD: 電場時間ステップ（GAINを大きくした時には小さくする）
- INITIAL_STATE: 初期状態
- TARGET_STATE: 目標状態
- EVALUATION_MODE: 評価関数のモード
- WEIGHT_MODE: 重みのモード
- CUSTOM_WEIGHTS: カスタム重み（例：V_max = 5で第4振動準位にポピュレーションを集めたい場合は[0,1,2,3,4,0]）

実行:
  python examples/example_local_optimization_arbitrary_system.py
"""

import os
import sys
import time
from typing import Optional, Union, Tuple, Literal

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


from utils.fft_utils import spectrogram_fast


# =========================
# Parameters
# =========================
# System parameters
V_MAX = 1  # 最大振動準位
WN_01 = 2349.1  # [cm^-1]
DELTA_WN = 25   # [cm^-1]
MU01 = 1e-30    # [C*m]
# Time grid
TIME_TOTAL = 1000.0  # [fs]
DT_EFIELD = 0.1      # [fs]
SAMPLE_STRIDE = 1    # 状態の履歴を保存するサンプリング間隔（SEGMENT_SIZE_STEPSの半分より小さくしてはいけない）

# Initial/Target
INITIAL_STATE = (0,)  # 初期状態
TARGET_STATE = (1,)   # 目標状態 (EVALUATION_MODE が "target" の時のみ最適電場の計算で使用される。使用の有無にかかわらず、フィデリティ計算では使用される。)

# 局所最適化の設定
GAIN = 1e21          # 局所制御則のゲイン（1e19-1e21程度が適切）
FIELD_MAX = 1e12      # [V/m] 電場の最大値。これを超える電場が設計された場合、電場はこの値にクリップされる。

# シード電場の設定
# 局所最適化では、初期状態でコヒーレンスがゼロだと、最適化計算が駆動されない。
# そこで、微小コヒーレンスを生成するために、パルス電場を与える
SEED_AMPLITUDE = 1e5  # シード電場の振幅
SEED_MAX_SEGMENTS = 5  # シードを許可する最大区間数
C_ABS_MIN = 1e-1  # 目標状態との重みがこれ以下ならシード電場を適用

# 評価関数の設定
#   mode: "target" uses A=|phi><phi| (ターゲットとの内積)
#         "weights" uses A=diag(w_i)（準位ごとの重み）
EVALUATION_MODE = "weights"  # "target" or "weights"
# EVALUATION_MODE = "target"  # "target" or "weights"
# WEIGHT_MODE options: "by_v_power", "custom", "by_v_power_reverse"
WEIGHT_MODE = "by_v_power"         
WEIGHT_V_POWER = 0.5          # by_v_power 用の指数（例: 2 で v^2）
WEIGHT_TARGET_FACTOR = 1  # ターゲット状態の重みを同じ振動準位のものより WEIGHT_TARGET_FACTOR 倍だけ大きくする
NORMALIZE_WEIGHTS = True  # 重みを正規化するかどうか
# カスタム重みの指定
CUSTOM_WEIGHTS = None # 基底次元に一致する長さの配列（list / np.ndarray）
# weightsモードで駆動量が極小のときのシード閾値
DRIVE_ABS_MIN = 1e-18

# 区間サイズの設定 (ステップ数または時間[fs]のいずれか)
# ステップ数が優先
# ここで指定したステップの時間幅の間で最適化電場を一定とする。
# 固有周波数の1/4より小さいくらいが適切
SEGMENT_SIZE_STEPS: Optional[int] = None
SEGMENT_SIZE_FS: Optional[float] = 0.5

SAVE_FIGURES = False


# =========================
# Utilities
# =========================

class Constants:
    C = 299792458  # m/s
    HBAR = 1.054571817e-34  # J*s

CONSTANTS = Constants()

def cm1_to_rad_phz(wn: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return wn * 2 * np.pi * CONSTANTS.C * 1e-13

def phz_to_cm1(phz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return phz / CONSTANTS.C * 1e13

def Cm_to_rad_phz_over_V_m(mu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return mu /(CONSTANTS.HBAR * 1e15)

class SystemConfig:
    def __init__(
        self,
        v_max: int,
        wn01: float,  # cm^-1
        delta_wn: float,  # cm^-1
        mu01: float,  # C*m
    ) -> None:
        self.v_max = v_max
        self.size = v_max + 1
        self.basis = [(i,) for i in range(v_max+1)]
        self.omega01 = cm1_to_rad_phz(wn01)
        self.delta_omega = cm1_to_rad_phz(delta_wn)
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}
        self.set_H0()
        self.set_mu(float(Cm_to_rad_phz_over_V_m(mu01)))
    def get_index(self, state: tuple[int, ...]) -> int:
        return self.index_map[state]
    def get_state(self, index: int) -> tuple[int, ...]:
        return self.basis[index]
    def set_initial_state(self, state: tuple[int, ...]) -> None:
        st = np.zeros(self.v_max+1, dtype=complex)
        st[self.get_index(state)] = 1.0
        self.initial_state = st
        self.initial_state_idx = self.get_index(state)
    def set_target_state(self, state: tuple[int, ...]) -> None:
        st = np.zeros(self.v_max+1, dtype=complex)
        st[self.get_index(state)] = 1.0
        self.target_state = st
        self.target_state_idx = self.get_index(state)
    def set_H0(self) -> None:
        vterm = np.arange(self.size) + 0.5
        self.eigenvalues = (self.omega01 + self.delta_omega) * vterm - self.delta_omega / 2 * vterm**2
        self.H0 = np.diag(self.eigenvalues)
    def set_mu(self, mu01: float) -> None:
        dipole_values = np.sqrt(np.arange(1, self.size))
        self.mu = mu01 * (np.diag(dipole_values, 1) + np.diag(dipole_values, -1))

def time_evolution(
    psi0: np.ndarray,
    H0: np.ndarray,
    mu: np.ndarray,
    E: np.ndarray,
    dt: float,
    stride:int,
    renorm:bool = False,
    ) -> np.ndarray:
    steps = (E.size - 1) // 2  # 必ず整数
    E3 = np.zeros((steps, 3), dtype=np.float64)
    E3[:, 0]  = E[0:-2:2]
    E3[:, 1] = E[1:-1:2]
    E3[:, 2] = E[2::2]
    
    if not isinstance(H0, csr_matrix):
        H0 = csr_matrix(H0)
    if not isinstance(mu, csr_matrix):
        mu = csr_matrix(mu)
    
    psi = psi0.copy()
    dim = psi.size
    n_out = steps // stride + 1
    out = np.empty((n_out, dim), dtype=np.complex128)
    out[0] = psi
    idx = 1

    # 1️⃣ 共通パターン（構造のみ）を作成
    pattern = ((H0 != 0) + (mu != 0))
    pattern = pattern.astype(np.complex128)  # 確実に複素数
    pattern.data[:] = 1.0 + 0j
    pattern = pattern.tocsr()

    # 2️⃣ パターンに合わせてデータを展開
    def expand_to_pattern(matrix: csr_matrix, pattern: csr_matrix) -> np.ndarray:
        result_data = np.zeros_like(pattern.data, dtype=np.complex128)
        m_csr = matrix.tocsr()
        pi, pj = pattern.nonzero()
        m_dict = {(i, j): v for i, j, v in zip(*m_csr.nonzero(), m_csr.data)}
        for idx_, (i, j) in enumerate(zip(pi, pj)):
            result_data[idx_] = m_dict.get((i, j), 0.0 + 0j)
        return result_data

    H0_data = expand_to_pattern(H0, pattern)
    mu_data = expand_to_pattern(mu, pattern)

    # 3️⃣ 計算用行列
    H = pattern.copy()

    # 4️⃣ 作業バッファ
    buf = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)

    for s in range(steps):
        e1, e2, e4 = E3[s]

        # H1
        H.data[:] = H0_data -mu_data * e1
        k1[:] = -1j * H.dot(psi)
        buf[:] = psi + 0.5 * dt * k1

        # H2
        H.data[:] = H0_data -mu_data * e2
        k2[:] = -1j * H.dot(buf)
        buf[:] = psi + 0.5 * dt * k2

        # H3
        H.data[:] = H0_data -mu_data * e2
        k3[:] = -1j * H.dot(buf)
        buf[:] = psi + dt * k3

        # H4
        H.data[:] = H0_data -mu_data * e4
        k4[:] = -1j * H.dot(buf)
        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        if renorm:
            # より高精度な正規化
            norm = np.sqrt((psi.conj() @ psi).real)
            if norm > 1e-12:  # 数値的にゼロでない場合のみ正規化
                psi *= 1.0 / norm
            else:
                continue
        
        if (s + 1) % stride == 0:
            out[idx] = psi
            idx += 1
    return out
    
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


def build_weights_for_basis(system: SystemConfig, *, mode: Literal['by_v_power', 'custom'],
                            normalize: bool = True,
                            custom: Optional[np.ndarray] = None,
                            v_power: float = 2.0,
                            reverse: bool = False) -> np.ndarray:
    """基底ごとの重みベクトル w_i を生成。
    - mode=="by_v": 各基底 (v,J,M) に対し w_i = v
    - mode=="by_v_power": w_i = v**v_power
    - mode=="custom": `custom` または `custom_dict` を使用
    - one_hot_target_idx が与えられた場合は、そのインデックスのみ1、他0
    正規化: max(w)>0 のとき w/=max(w)
    """
    dim = system.size
    if mode == "custom":
        if custom is not None:
            w = np.asarray(custom, dtype=float)
            if w.shape[0] != dim:
                raise ValueError(f"CUSTOM_WEIGHTS の長さ {w.shape[0]} が基底次元 {dim} と一致しません")
        else:
            raise ValueError("CUSTOM_WEIGHTS / CUSTOM_WEIGHTS_DICT のいずれも指定されていません")
    elif mode == "by_v_power":
        w = np.zeros(dim, dtype=float)
        for i, state in enumerate(system.basis):
            v = int(state[0])
            w[i] = float(v) ** float(v_power)
    if normalize:
        wmax = float(np.max(w))
        if wmax > 0:
            w = w / wmax
    # 逆順適用（one-hot時は無効）
    if reverse:
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

    def __init__(self, system: SystemConfig,
                 initial_idx: int, target_idx: int, time_total: float, dt: float,
                 sample_stride: int = 1,
                 seg_steps: Optional[int] = SEGMENT_SIZE_STEPS,
                 seg_fs: Optional[float] = SEGMENT_SIZE_FS,
                 ) -> None:
        self.system = system
        self.initial_idx = initial_idx
        self.target_idx = target_idx
        system.set_initial_state(self.system.basis[initial_idx])
        system.set_target_state(self.system.basis[target_idx])
        self.psi_initial = system.initial_state
        self.psi_target = system.target_state
        self.dt = float(dt)
        self.sample_stride = sample_stride
        self.segments, self.tlist = build_segments_and_tlist(time_total, dt, seg_steps, seg_fs)
        self.tlist_psi = self.tlist[::2*self.sample_stride]
        # RK4整合時間軸
        self.n_steps = len(self.tlist)
        self.n_traj_steps = self.n_steps // 2 // self.sample_stride

        
        # μ'（伝播単位系）
        self.mu = self.system.mu
        print(f"mu: {self.mu[0, 1]}")

        # 自由進行用の固有値（rad/fs）をキャッシュ
        try:
            self._eigenvalues = self.system.eigenvalues
            print(f"  固有値: {self._eigenvalues}")
        except Exception:
            self._eigenvalues = None

        # 評価演算子のセットアップ
        self.eval_mode = str(EVALUATION_MODE).lower()
        if self.eval_mode == "weights":
            mode_str = str(WEIGHT_MODE).lower()
            reverse_flag = mode_str.endswith("_reverse")
            base_mode = mode_str.replace("_reverse", "")
            self.A_diag = build_weights_for_basis(
                self.system,
                mode=base_mode,
                normalize=bool(NORMALIZE_WEIGHTS),
                custom=np.asarray(CUSTOM_WEIGHTS, dtype=float) if CUSTOM_WEIGHTS is not None else None,
                v_power=float(WEIGHT_V_POWER),
                reverse=reverse_flag,
            )
            # ターゲット状態の重みを同じ振動順位のものより WEIGHT_TARGET_FACTOR 倍だけ大きくする
            v_target = int(self.system.basis[self.target_idx][0])
            w_before = float(self.A_diag[self.target_idx])
            self.A_diag[self.target_idx] = w_before * float(WEIGHT_TARGET_FACTOR)
            print(f"  ターゲット重み強調: idx={self.target_idx}, v={v_target}, w={w_before:.3g} -> {float(self.A_diag[self.target_idx]):.3g} (x{WEIGHT_TARGET_FACTOR})")
            print("評価モード: weights (対角重み)")
            print(f"  重み min/max = {float(np.min(self.A_diag)):.3g}/{float(np.max(self.A_diag)):.3g}")
            if reverse_flag:
                print("  注記: 逆順重み（reverse）を適用しています")
            # 重みの要約を表示（上位/下位を少数表示）
            try:
                idx_sorted = np.argsort(self.A_diag)
                show = min(5, len(idx_sorted))
                print("  上位重み: ")
                for k in idx_sorted[::-1][:show]:
                    print(f"    idx={k}, state={self.system.basis[k]}, w={self.A_diag[k]:.3g}")
                print("  下位重み: ")
                for k in idx_sorted[:show]:
                    print(f"    idx={k}, state={self.system.basis[k]}, w={self.A_diag[k]:.3g}")
            except Exception:
                pass
        else:
            # target projector mode uses A = |phi><phi|（内部で専用式を使用）
            self.A_diag = None
            print("評価モード: target (|phi><phi|)")

        print("初期化完了:")
        print(f"  基底次元: {self.system.size}")
        print(f"  電場時間ステップ数: {self.n_steps}")
        print(f"  軌跡配列長(予想): {self.n_traj_steps}")
        print(f"  区間数: {len(self.segments)}")
        print(f"  区間長: {self.segments[0][1] - self.segments[0][0]} 点 ({self.dt * (self.segments[0][1] - self.segments[0][0])} fs)")
        print(f"  初期状態: {self.system.basis[initial_idx]} → idx={initial_idx}")
        print(f"  目標状態: {self.system.basis[target_idx]} → idx={target_idx}")

    def evolve(self, psi0: np.ndarray, e: np.ndarray) -> np.ndarray:
        """単一区間の定数電場で伝播し、時間と軌跡を返す。"""
        return time_evolution(
            psi0=psi0,
            H0=self.system.H0,
            mu=self.system.mu,
            E=e,
            dt=self.dt*2,
            stride=self.sample_stride,
            renorm=True,
        )

    def design_field(self,
                     gain: float = GAIN,
                    ) -> np.ndarray:
        """区間ごとに定数電場を設計し、全体の ElectricField を返す。"""
        segments = self.segments
        full_field = np.zeros((self.n_steps, ), dtype=float)

        psi_curr = self.psi_initial.copy()
        t_curr_idx = 0
        seed_left = SEED_MAX_SEGMENTS

        for (start, end) in segments:
            
            # 局所制御則（モード切替）
            if self.eval_mode == "weights" and self.A_diag is not None:
                # A = diag(w) のとき、E_a ∝ Im⟨ψ|[A, μ_a']|ψ⟩
                A_diag = self.A_diag
                # term1 = <ψ|A μ ψ>, term2 = <ψ| μ A ψ>
                mu_psi = -self.system.mu @ psi_curr
                A_mu_psi = A_diag * mu_psi
                term = float(np.imag(complex(np.vdot(psi_curr, A_mu_psi))))
                e = float(gain * term)
                # 駆動が極小のときシード
                if (abs(term) < DRIVE_ABS_MIN and seed_left > 0):
                    e = SEED_AMPLITUDE
                    seed_left -= 1
                    print(f"[seed] 区間[{start}:{end}] weights駆動が極小のためシード: E={e:.3e}")
            else:
                # target projector モード
                c = complex(np.vdot(self.psi_target, psi_curr))
                d = complex(np.vdot(self.psi_target, (-self.mu @ psi_curr)))
                val = float(np.imag(np.conj(c) * d))
                e = float(gain * val)
                if abs(c) < C_ABS_MIN and seed_left > 0:
                    s = 1.0 if (abs(d) == 0.0) else (1.0 if (np.real(d) >= 0.0) else -1.0)
                    e = s * SEED_AMPLITUDE
                    seed_left -= 1
                    print(f"[seed] 区間[{start}:{end}] にシード電場を適用: E={e:.3e}, |c|={abs(c):.3e}")

            # 振幅制限
            e = float(np.clip(e, -FIELD_MAX, FIELD_MAX))

            # 区間フィールドをセット
            full_field[start+1:end+1] = e

            psi_traj_seg = self.evolve(psi_curr, full_field[start:end+1])
            psi_curr = psi_traj_seg[-1]

        # ランニングコスト J_a = 1/2 * λ * ∫ S(t) |E|^2 dt（表示用）
        field_penalty = 1/gain
        try:
            E2 = full_field ** 2
            self.running_cost = float(field_penalty) * float(np.sum(E2) * self.dt)
        except Exception:
            self.running_cost = None
        return full_field

def spectrogram_fast(
    x: np.ndarray,
    y: np.ndarray,
    T: Union[int, float],
    unit_T: Literal['index', 'x'] = 'index',
    window_type: Literal['rectangle', 'triangle', 'hamming', 'han', 'blackman'] = 'hamming',
    return_max_index: bool = False,
    step: int = 1,
    N_pad: Optional[int] = None,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    dx = x[1] - x[0]
    N = len(x)

    if unit_T == 'x':
        T_index = int(np.round(T / dx))
    elif unit_T == 'index':
        T_index = int(T)
    else:
        raise ValueError("unit_T must be 'index' or 'x'")

    if T_index > N:
        raise ValueError("Window size T is larger than input signal.")

    if window_type == 'rectangle':
        window = np.ones(T_index)
    elif window_type == 'triangle':
        window = 1 - np.abs((np.arange(T_index) - T_index / 2) / (T_index / 2))
    elif window_type == 'hamming':
        window = np.hamming(T_index)
    elif window_type == 'han':
        window = np.hanning(T_index)
    elif window_type == 'blackman':
        window = np.blackman(T_index)
    else:
        raise ValueError("Unknown window type")

    # Use rFFT with zero-padding length N_pad if provided
    n_rfft = int(T_index if N_pad is None else max(int(N_pad), int(T_index)))
    freq = np.fft.rfftfreq(n_rfft, d=dx)
    num_freq = len(freq)

    num_windows = (N - T_index) // step + 1
    spec = np.empty((num_freq, num_windows), dtype=np.float64)
    max_indices = np.empty(num_windows, dtype=int) if return_max_index else None
    x_spec = np.empty(num_windows, dtype=np.float64)

    for i in range(num_windows):
        start = i * step
        end = start + T_index
        segment = y[start:end] * window
        # rFFT with zero-padding to n_rfft
        fft_segment = np.fft.rfft(segment, n=n_rfft)
        abs_fft = np.abs(fft_segment)
        spec[:, i] = abs_fft
        if return_max_index and max_indices is not None:
            max_indices[i] = np.argmax(abs_fft)
        x_spec[i] = x[start + T_index // 2]

    if return_max_index:
        if max_indices is None:
            max_indices = np.array([], dtype=int)
        return x_spec, freq, spec, max_indices
    else:
        return x_spec, freq, spec




if __name__ == "__main__":
    print("=== 局所最適化理論による振動回転励起 電場設計 ===")
    print(f"基底設定: V_max={V_MAX}")

    system = SystemConfig(
        v_max=V_MAX,
        wn01=WN_01,
        delta_wn=DELTA_WN,
        mu01=MU01,
    )



    initial_idx = system.get_index(INITIAL_STATE)
    target_idx = system.get_index(TARGET_STATE)
    print(f"初期状態: {INITIAL_STATE} (idx={initial_idx})")
    print(f"目標状態: {TARGET_STATE} (idx={target_idx})")

    optimizer = LocalOptimizer(
        system=system,
        initial_idx=initial_idx,
        target_idx=target_idx,
        time_total=TIME_TOTAL,
        dt=DT_EFIELD,
        sample_stride=SAMPLE_STRIDE,
    )

    start = time.time()
    efield = optimizer.design_field()
    elapsed = time.time() - start
    print(f"設計完了: {elapsed:.2f} s")
    if hasattr(optimizer, 'running_cost') and optimizer.running_cost is not None:
        print(f"ランニングコスト J_a (λ/2 ∫ S|E|^2 dt): {optimizer.running_cost:.6e}")

    # 最終評価（全区間をまとめて前進）
    psi_traj = optimizer.evolve(optimizer.psi_initial, efield)
    fidelity = calculate_fidelity(psi_traj[-1], optimizer.target_idx)
    print(f"最終フィデリティー: {fidelity:.6f}")

    # %% Plotting
    prob = np.abs(psi_traj) ** 2
    t = optimizer.tlist
    t_psi = optimizer.tlist_psi
    # スペクトログラム用の周波数範囲の初期値（波数 cm^-1）
    fmin: float = float(max(0.0, WN_01 - 500.0))
    fmax: float = float(WN_01 + 500.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Field
    axes[0, 0].plot(t, efield, 'r-', label='E(t)')
    axes[0, 0].set_xlabel('Time [fs]')
    axes[0, 0].set_ylabel('Electric Field [V/m]')
    axes[0, 0].set_title('Designed Electric Field (Local Optimization)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # axes[0, 0].set_xlim(990, 1000)

    # 2) Fidelity (instantaneous target population)
    idx_tar = optimizer.target_idx
    axes[0, 1].plot(t_psi, prob[:, idx_tar], 'g-')
    axes[0, 1].set_xlabel('Time [fs]')
    axes[0, 1].set_ylabel('Population of target')
    axes[0, 1].set_title('Target Population vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # 3) Populations (highlight initial/target)

    highlight_states = [INITIAL_STATE, TARGET_STATE]
    highlight_set = {tuple(s) for s in highlight_states}
    for state in highlight_states:
        if state in optimizer.system.index_map:
            idx = optimizer.system.get_index(state)
            label = f'|v={state[0]}⟩ (highlight)'
            axes[1, 0].plot(t_psi, prob[:, idx], linewidth=2.5, label=label)
    for i, st in enumerate(optimizer.system.basis):
        if st in highlight_set:
            continue
        label = f'|v={st[0]}⟩'
        axes[1, 0].plot(t_psi, prob[:, i], linewidth=1.0, alpha=0.9, label=label)
    axes[1, 0].set_xlabel('Time [fs]')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_title('State Population Evolution')
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)

    # 4) Field norm (per segment view is implicit); show overall norm
    field_norm = np.linalg.norm(efield)
    axes[1, 1].plot([0, len(t)-1], [field_norm, field_norm], 'k-')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('||E||')
    axes[1, 1].set_title('Field Norm (overall)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure under examples/figures [[memory:2714886]]
    if SAVE_FIGURES:
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"local_optimization_results_{timestamp}.png"
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
    plt.show()
    # %%
    # ==========================================
    # Spectrum of designed field (wavenumber cm^-1)
    # - zero padding to ~0.1 cm^-1 resolution
    # - 1/e^2 intensity window around the strongest line (symmetric)
    # - linear phase removal at time center and slope removal at peak
    # - overlay R(J)_V and P(J)_V transitions from H0 and selection rules
    # ==========================================
    # Npad は後段のスペクトログラム処理でも使用するため既定値で初期化
    Npad = 0
    wn_lim = [1900, 3000]
    try:
        N = len(t)

        # target resolution 0.1 cm^-1 → PHz
        df_target_PHz = float(cm1_to_rad_phz(1)/(2*np.pi))
        Npad = int(np.ceil(1.0 / (optimizer.dt * df_target_PHz)))
        print(f"Npad: {Npad}, N: {N}")
        Npad = max(Npad, N)

        E_freq = np.fft.rfft(efield, n=Npad, axis=0)
        freq_PHz = np.fft.rfftfreq(Npad, d=optimizer.dt)
        freq_cm = np.asarray(freq_PHz/CONSTANTS.C*1e13, dtype=float)

        t_center = optimizer.tlist[-1] / 2.0
        E_freq_comp = E_freq * (np.exp(1j * 2 * np.pi * freq_PHz * t_center))
        intensity = np.abs(E_freq_comp) ** 2
        peak_idx = int(np.argmax(intensity))
        f0 = float(freq_cm[peak_idx])
        eigenvalues = optimizer.system.eigenvalues
        states = optimizer.system.basis
        trans_wn: list[float] = []
        for v in range(optimizer.system.v_max):
            wn = float((eigenvalues[v+1] - eigenvalues[v]) / CONSTANTS.C * 1e13)
            if np.isfinite(wn) and wn > 0:
                trans_wn.append(wn)
        trans_wn = np.array(trans_wn)
        if len(trans_wn) > 1:
            wn_min = float(np.min(trans_wn))
            wn_max = float(np.max(trans_wn))
            center = 0.5 * (wn_min + wn_max)
            span = max(wn_max - wn_min, 1e-6)
            factor = 5
            half = 0.5 * span * factor
            fmin = max(center - half, float(freq_cm[0]))
            fmax = min(center + half, float(freq_cm[-1]))
        else:
            fmin = max(0.0, float(WN_01) - 500.0)
            fmax = float(WN_01) + 500.0
        if wn_lim is not None:
            fmin = wn_lim[0]
            fmax = wn_lim[1]
        phase_raw = np.unwrap(np.angle(E_freq_comp))
        print("ここまでok1")
        dphidk = np.gradient(phase_raw, freq_cm)
        slope = float(dphidk[peak_idx])
        phase = phase_raw - (slope * (freq_cm - f0) + phase_raw[peak_idx])
        mask = (freq_cm >= fmin) & (freq_cm <= fmax)
        freq_p = freq_cm[mask]
        intensity_p = intensity[mask]
        phase_p = phase[mask]
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot(freq_p, intensity_p, color='tab:blue', label='|E|²')
        ax.set_ylabel('Intensity (a.u.)')
        # ax0.grid(True, alpha=0.3)
        ax_t = ax.twinx()
        ax_t.plot(freq_p, phase_p, color='tab:red', alpha=0.7, label='Phase')
        ax_t.set_ylabel('Phase (rad)')
        ax.set_title('Designed Field Spectrum (E)')
        ax.set_xlim(fmin, fmax)
        lines0, labels0 = ax.get_legend_handles_labels()
        lines0_t, labels0_t = ax_t.get_legend_handles_labels()
        ax.legend(lines0 + lines0_t, labels0 + labels0_t, loc='upper right')

        try:
            eigenvalues = optimizer.system.eigenvalues
            states = optimizer.system.basis
            lines_v: list[tuple[float, str]] = [(eigenvalues[v+1] - eigenvalues[v], f"{v}→{v+1}") for v in range(optimizer.system.v_max)]
            y0 = float(np.max(intensity_p)) if intensity_p.size else 1.0
            for wn, lbl in lines_v:
                if fmin <= wn <= fmax:
                    ax.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                    ax.text(wn, y0, lbl, rotation=-90, va='top', ha='center', fontsize=8, color='gray')
        except Exception as e_lines:
            print(f"遷移線オーバーレイでエラー: {e_lines}")

        plt.tight_layout()
        # Save spectrum figure [[memory:2714886]]
        if SAVE_FIGURES:
            figures_dir = os.path.join(os.path.dirname(__file__), "figures")
            os.makedirs(figures_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename2 = f"local_optimization_spectrum_{timestamp}.png"
            filepath2 = os.path.join(figures_dir, filename2)
            plt.savefig(filepath2, dpi=300, bbox_inches='tight')
            print(f"スペクトル図を保存しました: {filepath2}")
        plt.show()
    except Exception as e:
        print(f"スペクトル可視化でエラー: {e}")
        # スペクトログラムのための周波数範囲が未定義の場合に備えて既定値を用意
        if 'fmin' not in locals():
            fmin = float(max(0.0, WN_01 - 500.0))
        if 'fmax' not in locals():
            fmax = float(WN_01 + 500.0)
        if 'Npad' not in locals():
            Npad = 0

    # ==========================================
    # Spectrogram using fft_utils.spectrogram_fast
    # - Use Ex(t) as representative
    # - Frequency axis in cm^-1, limited to [fmin, fmax] from spectrum
    # ==========================================
    try:
        # fmin/fmax が未定義のときは安全な既定値を設定
        if 'fmin' not in locals() or 'fmax' not in locals():
            # スペクトル中心を WN_01 付近と仮定して広めに設定
            center = float(WN_01)
            span = 500.0
            fmin = center - span
            fmax = center + span
        # 窓幅は全長の一部を設定（例: 2048 サンプル相当）
        T_index = len(t) // 20
        res = spectrogram_fast(t, efield, T=T_index, unit_T='index', window_type='hamming', step=max(1, T_index // 8), N_pad=Npad)
        if len(res) == 4:
            x_spec, freq_1fs, spec, _max_idx = res
        else:
            x_spec, freq_1fs, spec = res
        # 周波数 [1/fs] → 波数[cm^-1]
        freq_cm_full = np.asarray(phz_to_cm1(freq_1fs))
        # 範囲をスペクトルの [fmin, fmax] に合わせる
        mask_rng = (freq_cm_full >= fmin) & (freq_cm_full <= fmax)
        freq_cm_plot = freq_cm_full[mask_rng]
        spec_plot = spec[mask_rng, :]

        # 描画
        X, Y = np.meshgrid(x_spec, freq_cm_plot)
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
        cf = ax3.pcolormesh(X, Y, spec_plot, shading='auto', cmap='viridis')
        ax3.set_xlabel('Time [fs]')
        ax3.set_ylabel('Wavenumber (cm$^{-1}$)')
        ax3.set_title('Spectrogram (E)')
        ax3.set_ylim(fmin, fmax)
        fig3.colorbar(cf, ax=ax3, label='|FFT|')
        plt.tight_layout()
        # 保存 [[memory:2714886]]
        if SAVE_FIGURES:
            figures_dir = os.path.join(os.path.dirname(__file__), "figures")
            os.makedirs(figures_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename3 = f"local_optimization_spectrogram_{timestamp}.png"
            filepath3 = os.path.join(figures_dir, filename3)
            plt.savefig(filepath3, dpi=300, bbox_inches='tight')
            print(f"スペクトログラム図を保存しました: {filepath3}")
        plt.show()
    except Exception as e:
        print(f"スペクトログラム可視化でエラー: {e}")