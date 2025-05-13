# params_template.py
import numpy as np

description = "pump_probe_example"

# === 時間設定 ===
t_start = -100
t_end = 100
t_points = 1000
def make_tlist(delay):
    return np.linspace(t_start, t_end, t_points)

# === 電場波形の設定 ===
gauss_widths = [20.0, 30.0]                # 包絡線の幅（fs）
polarizations = [[1, 0], [1, 1j]]         # ジョーンズベクトル
carrier_freq = 0.2 * np.pi               # 中心周波数
amplitude = 1.0

# 群遅延分散など
gdd = 0.0
tod = 0.0
delays = [0.0, 20.0, 40.0]                # パルス遅延（fs）

# === 系のパラメータ ===
V_max = 1
J_max = 1
h = 1.0
omega = 0.1 * np.pi
B = 0.02
alpha = 0.001
delta_omega = 0.005 * np.pi
