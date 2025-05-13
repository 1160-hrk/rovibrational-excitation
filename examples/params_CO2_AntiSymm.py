# params_CO2_antisymm.py
import numpy as np

description = "CO2_antisymm_stretch"

# === 時間設定 ===
t_start = -200
t_end = 200
t_points = 2000
def make_tlist(delay):
    return np.linspace(t_start, t_end, t_points)

# === 電場波形の設定 ===
gauss_widths = [50.0, 80.0]                  # fs
polarizations = [[1, 0], [1/np.sqrt(2), 1j/np.sqrt(2)]]  # 直線偏光, 円偏光
carrier_freq = 2349.0 * 2 * np.pi * 1e12 * 1e-15  # rad/fs, 2349 cm^-1 → CO2 非対称伸縮振動の赤外線吸収周波数
amplitude = 1.0

gdd = 0.0
tod = 0.0
delays = [0.0, 100.0, 200.0]                 # fs（遅延時間）

# === 系のパラメータ（CO2 非対称振動） ===
V_max = 2
J_max = 2
h = 1.0
omega = 2349.0 * 2 * np.pi * 1e12 * 1e-15    # rad/fs（基準）
B = 0.3902 * 1e-3 * 2 * np.pi * 1e12 * 1e-15   # rad/fs（CO2の回転定数、単位変換済）
alpha = 0.0                                   # CO2 非対称振動に対して小さいので 0 近似
debye_unit = 3.33564e-30  # C*m
# CO2の電気双極子モーメント（単位変換済み）
dipole_constant = 0.3 * debye_unit  # C*m
# 非調和性の補正（仮設定。詳細不明な場合は小さく）
delta_omega = 10.0 * 2 * np.pi * 1e12 * 1e-15