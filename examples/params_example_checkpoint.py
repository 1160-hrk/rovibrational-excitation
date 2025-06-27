#!/usr/bin/env python
"""
チェックポイント機能デモ用パラメータファイル
==================================
長時間計算をシミュレートするためのパラメータ設定例
"""
import numpy as np

# 計算の説明（結果ディレクトリ名に使用）
description = "checkpoint_demo"

# === 時間軸設定 ===
t_start, t_end, dt = -100.0, 100.0, 0.1  # [fs]

# === 基本パラメータ（固定） ===
V_max, J_max = 2, 2                          # 計算を軽くするため小さく設定
omega_rad_phz = 2349 * 2 * np.pi * 3e10 / 1e15   # 振動周波数 [rad/fs]
mu0_Cm = 0.3 * 3.33564e-30                  # 双極子モーメント [C·m]
t_center = 0.0                               # パルス中心時刻 [fs]
carrier_freq = omega_rad_phz                 # キャリア周波数 [rad/fs]

# === スイープパラメータ ===
# 複数の値を指定すると自動的にスイープ対象になる
duration = [30.0, 40.0, 50.0, 60.0, 70.0]   # パルス幅 [fs] - 5ケース
amplitude = [1e8, 5e8, 1e9, 5e9, 1e10]      # 電場強度 [V/m] - 5ケース
polarization = [
    [1, 0],           # x偏光
    [0, 1],           # y偏光  
    [1/np.sqrt(2), 1j/np.sqrt(2)]  # 円偏光
]                                            # 偏光 - 3ケース

# 合計ケース数: 5 × 5 × 3 = 75ケース

# === 計算設定 ===
sample_stride = 5        # サンプリング間隔（計算高速化のため）
backend = "numpy"        # または "cupy"（GPU使用時） 