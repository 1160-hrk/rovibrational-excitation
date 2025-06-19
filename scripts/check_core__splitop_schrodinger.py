# check_core__splitop_schrodinger.py
# ---------------------------------------------------------------
# split-operator 版の動作確認スクリプト
#
#   * 2-レベル系（ω₀=1）を x 偏光 Gaussian パルスで 1 Rabi 周期だけ励起
#   * CPU / NumPy 実行
#   * rk4 用チェックスクリプト（check_core__rk4_schrodinger.py）と
#     ほぼ同じ可視化が得られる
#
# 使い方:
#   $ python check_core__splitop_schrodinger.py
# ---------------------------------------------------------------

import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

# カレント → ../src を import パスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core._splitop_schrodinger import splitop_schrodinger

# ----------------------------------------------------------------------
# 1. シミュレーション条件
# ----------------------------------------------------------------------
ti, tf          = 0.0, 1000000.0
dt_E            = 0.01                # 電場サンプリング幅
dt              = dt_E * 2            # ψ を進める dt (=2*dt_E)
sample_stride   = 1

t_E  = np.arange(ti, tf + 3*dt_E, dt_E)     # 電場サンプル時刻
t_psi= t_E[::2][::sample_stride]            # ψ を出力する時刻

omega0 = 1.0
H0 = np.array([[0.0,     0.0],
               [0.0, omega0]], dtype=np.complex128)

mu_x = np.array([[0.0,  1.0],
                 [1.0,  0.0]], dtype=np.complex128)

mu_y = np.array([[0.0,     1.0j],
                 [-1.0j,   0.0 ]], dtype=np.complex128)

# --- Gaussian パルス（FWHM ≈ 2.355*duration）
duration = 10000.0
tc       = 0.5*(t_E[0] + t_E[-1])
env      = np.exp(-(t_E-tc)**2 / (2*duration**2))

num_rabi = 1
amp      = num_rabi / (duration*np.sqrt(2*np.pi)) * 2*np.pi     # 1 Rabi 周期
Efield   = amp * env * np.cos(omega0*(t_E-tc))                  # 実電場
Efield   = Efield.astype(np.float64)

# 偏光ベクトル p = [1,0]  (x 偏光, 複素型で与える)
pol = np.array([1.0+0.0j, 0.0+0.0j], dtype=np.complex128)
pol = np.array([1.0+0.0j, 0.0+1.0j], dtype=np.complex128)

# 初期状態 |ψ(0)⟩ = (1,0)ᵀ
psi0 = np.array([1.0+0.0j, 0.0+0.0j], dtype=np.complex128)

# ステップ数は (len(Efield)-1)//2
steps = (len(Efield)-1)//2

# ----------------------------------------------------------------------
# 2. 伝搬
# ----------------------------------------------------------------------
print("propagating …")
t0 = time.time()
psi_traj = splitop_schrodinger(
    H0,
    mu_x,
    mu_y,
    pol,
    Efield,
    psi0,
    dt,
    steps,
    sample_stride=sample_stride,
)
print(f"done   (elapsed {time.time()-t0:.2f} s)")

# ----------------------------------------------------------------------
# 3. 可視化
# ----------------------------------------------------------------------
fig, ax = plt.subplots(
    2, 1, figsize=(6, 6),
    sharex=True, gridspec_kw=dict(hspace=0.0, height_ratios=[0.2, 1])
)

ax[0].plot(t_E, Efield, color="tab:gray")
ax[0].set_ylabel("E(t)")

ax[1].plot(t_psi, np.abs(psi_traj[:, 0])**2, label=r"|ψ₀|²")
ax[1].plot(t_psi, np.abs(psi_traj[:, 1])**2, label=r"|ψ₁|²")
ax[1].plot(t_psi, (np.abs(psi_traj)**2).sum(axis=1), label=r"|ψ|²", lw=0.7)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Population")
ax[1].legend(loc="upper right")

plt.show()
