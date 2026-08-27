#!/usr/bin/env python
"""
新しいスイープ仕様のデモ用パラメータファイル
======================================
推奨案1（専用キーワードでスイープ制御）の使用例
"""

# 計算の説明（結果ディレクトリ名に使用）
description = "new_sweep_demo"

# === 時間軸設定 ===
t_start, t_end, dt = -50.0, 50.0, 0.1  # [fs]

# === 基本パラメータ（固定） ===
V_max, J_max = 2, 2  # 量子数上限
vibrational_frequency = 2349.0
vibrational_frequency_units = "cm^-1"
anharmonic_shift = 0.0
anharmonic_shift_units = "cm^-1"
rotational_constant = 0.3902
rotational_constant_units = "cm^-1"
vibration_rotation_coupling = 0.0
vibration_rotation_coupling_units = "cm^-1"
mu0_Cm = 0.3 * 3.33564e-30  # 双極子モーメント [C·m] (~0.3 Debye)
t_center = 0.0  # パルス中心時刻 [fs]
carrier_frequency = 2349.0  # キャリア波数
carrier_frequency_units = "cm^-1"

# === 新しいスイープ仕様 ===

# 1. 通常のリスト → スイープ対象（従来通り）
duration = [20.0, 30.0, 40.0]  # パルス幅 [fs] - 3ケース

# 2. FIXED_VALUE_KEYS → 常に固定値
polarization = [1.0, 0.0]  # x偏光（常に固定値）
# 複数偏光をスイープしたい場合は：
# polarization_sweep = [[1, 0], [0, 1], [1/√2, 1j/√2]]

# 3. _sweep接尾辞 → 明示的にスイープ対象（接尾辞は自動的に除去される）
amplitude_sweep = [1e8, 5e8, 1e9]  # 電場強度 [V/m] - 3ケース → 'amplitude' として保存

# 4. 単一要素リスト → 固定値（従来通り）
initial_states = [0]  # 初期状態（基底状態のみ）

# 合計ケース数: 3（duration） × 3（amplitude_sweep→amplitude） = 9ケース
# polarizationは固定値なので影響しない
# 各ケースでは params["amplitude"] と params["duration"] でアクセス可能

# === その他の設定 ===
sample_stride = 2  # サンプリング間隔
backend = "numpy"  # 計算バックエンド
