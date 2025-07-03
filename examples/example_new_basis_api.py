#!/usr/bin/env python3
"""
新しい基底APIのデモンストレーション

物理パラメータを基底クラスに統合することで、より直感的で使いやすいAPIを実現しました。
"""

import numpy as np
import matplotlib.pyplot as plt
from rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis, VibLadderBasis

print("=== 新しい基底APIのデモ ===\n")

# ========================================
# 1. CO2分子の例（実際の分子パラメータ）
# ========================================
print("1. CO2分子のν3モード（反対称伸縮振動）")
print("-" * 50)

# 新API: 全てのパラメータを基底作成時に指定
co2_basis = LinMolBasis(
    V_max=3,           # 振動準位 v=0,1,2,3
    J_max=20,          # 回転準位 J=0,1,...,20
    omega=2349,        # ν3モード周波数 [cm⁻¹]
    B=0.3902,          # 回転定数 [cm⁻¹]
    alpha=0.0,         # 振動回転相互作用（今回は無視）
    delta_omega=0.0,   # 非調和性（今回は無視）
    input_units="cm^-1",
    output_units="J"
)

print(f"基底情報: {co2_basis}")
print(f"基底の次元: {co2_basis.size()}")

# ハミルトニアン生成（引数不要！）
H0 = co2_basis.generate_H0()
print(f"\nハミルトニアン: {H0}")

# エネルギー準位を表示
eigvals = H0.eigenvalues[:10]  # 最初の10個
print("\n最低10準位のエネルギー [J]:")
for i, E in enumerate(eigvals):
    print(f"  {i}: {E:.3e} J")

# ========================================
# 2. 異なる単位系での入力例
# ========================================
print("\n\n2. 様々な単位系での入力")
print("-" * 50)

# THz単位での入力
basis_thz = VibLadderBasis(
    V_max=5,
    omega=10.0,         # 10 THz
    delta_omega=0.1,    # 0.1 THz anharmonicity
    input_units="THz",
    output_units="rad/fs"
)
print(f"THz入力: {basis_thz}")

# eV単位での入力
basis_ev = TwoLevelBasis(
    energy_gap=0.5,     # 0.5 eV
    input_units="eV",
    output_units="J"
)
print(f"eV入力: {basis_ev}")

# ========================================
# 3. パラメータスキャン（後方互換性）
# ========================================
print("\n\n3. パラメータスキャンの例")
print("-" * 50)

# 回転定数を変えてエネルギー差を計算
B_values = np.linspace(0.3, 0.5, 5)  # cm⁻¹
energy_gaps = []

for B in B_values:
    # generate_H0_with_paramsで一時的にパラメータを変更
    H_temp = co2_basis.generate_H0_with_params(B=B, input_units="cm^-1")
    gap = H_temp.eigenvalues[1] - H_temp.eigenvalues[0]
    energy_gaps.append(gap)
    print(f"B = {B:.3f} cm⁻¹ → ΔE = {gap:.3e} J")

# ========================================
# 4. 物理的に意味のある計算例
# ========================================
print("\n\n4. 振動回転スペクトルの計算")
print("-" * 50)

# より現実的なCO2パラメータ
co2_realistic = LinMolBasis(
    V_max=1,            # v=0,1のみ
    J_max=30,           # J=0-30
    use_M=False,        # M依存性は無視
    omega=2349.14,      # より正確な値
    B=0.39021,          # より正確な値
    alpha=0.0,          # 簡単のため無視
    delta_omega=0.0,    # 簡単のため無視
    input_units="cm^-1",
    output_units="cm^-1"  # スペクトル解析用にcm⁻¹で出力
)

H0_cm = co2_realistic.generate_H0()
energies = H0_cm.eigenvalues

# P枝とR枝の遷移を計算
print("P枝遷移 (ΔJ = -1):")
for J in range(1, 6):
    # |v=0,J⟩ → |v=1,J-1⟩
    idx_lower = co2_realistic.get_index([0, J])
    idx_upper = co2_realistic.get_index([1, J-1])
    transition = energies[idx_upper] - energies[idx_lower]
    print(f"  P({J}): {transition:.2f} cm⁻¹")

print("\nR枝遷移 (ΔJ = +1):")
for J in range(0, 5):
    # |v=0,J⟩ → |v=1,J+1⟩
    idx_lower = co2_realistic.get_index([0, J])
    idx_upper = co2_realistic.get_index([1, J+1])
    transition = energies[idx_upper] - energies[idx_lower]
    print(f"  R({J}): {transition:.2f} cm⁻¹")

print("\n✅ デモンストレーション完了！")
print("\n新しいAPIの利点:")
print("- 物理パラメータを基底に統合 → 一貫性の向上")
print("- 任意の単位系をサポート → 使いやすさの向上")
print("- generate_H0()が引数不要 → シンプルなインターフェース")
print("- 後方互換性を維持 → 既存コードも動作") 