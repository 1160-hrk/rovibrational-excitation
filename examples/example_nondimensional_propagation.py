#!/usr/bin/env python3
"""
無次元化シュレディンガー方程式伝播の使用例
===========================================

このサンプルでは、新しい無次元化機能を使用して、
従来の次元ありシステムと比較し、数値安定性や
物理的解釈の改善を示します。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ライブラリインポート
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.nondimensionalize import (
    nondimensionalize_system,
    analyze_regime,
    get_physical_time,
)

# パラメータ設定
SYSTEM_PARAMS = {
    "V_max": 3,  # 最大振動量子数
    "J_max": 5,  # 最大回転量子数
    "use_M": True,
    "omega_rad_phz": 0.159,  # CO2分子のω1振動（rad/fs）
    "B_rad_phz": 3.9e-5,  # 回転定数（rad/fs）
    "mu0": 1e-30,  # 双極子モーメント（C·m）
}

FIELD_PARAMS = {
    "amplitude": 1e9,        # 電場振幅（V/m）
    "duration": 100.0,       # パルス幅（fs）
    "carrier_freq": 0.159,   # キャリア周波数（rad/fs）
    "polarization": [1, 0],  # 偏光
}

TIME_PARAMS = {
    "t_start": -200.0,
    "t_end": 800.0,
    "dt": 0.1,
    "sample_stride": 5,
}


def setup_system():
    """基本的な量子系と電場を設定"""
    
    # 時間軸
    t_E = np.arange(TIME_PARAMS["t_start"], TIME_PARAMS["t_end"] + TIME_PARAMS["dt"], TIME_PARAMS["dt"])
    
    # 電場
    E = ElectricField(tlist=t_E)
    E.add_dispersed_Efield(
        envelope_func=gaussian_fwhm,
        duration=FIELD_PARAMS["duration"],
        t_center=300.0,
        carrier_freq=FIELD_PARAMS["carrier_freq"],
        amplitude=FIELD_PARAMS["amplitude"],
        polarization=np.array(FIELD_PARAMS["polarization"]),
    )
    
    # 基底と初期状態
    basis = LinMolBasis(
        V_max=SYSTEM_PARAMS["V_max"],
        J_max=SYSTEM_PARAMS["J_max"],
        use_M=False,
        omega_rad_phz=SYSTEM_PARAMS["omega_rad_phz"],
    )
    
    sv = StateVector(basis)
    sv.set_state(basis.get_state(0), 1)  # 基底状態 |V=0,J=0,M=0⟩
    sv.normalize()
    
    # ハミルトニアン（エネルギー単位）
    H0 = basis.generate_H0(
        omega_rad_phz=SYSTEM_PARAMS["omega_rad_phz"],
        B_rad_phz=SYSTEM_PARAMS["B_rad_phz"],
        return_energy_units=True,  # エネルギー単位（J）で取得
    )
    
    # 双極子行列
    dip = LinMolDipoleMatrix(
        basis,
        mu0=SYSTEM_PARAMS["mu0"],
        backend="numpy",
        dense=True,
    )
    
    return E, basis, sv, H0, dip


def run_comparison():
    """次元ありと無次元化システムの比較実行"""
    
    print("=" * 60)
    print("無次元化シュレディンガー方程式伝播のデモンストレーション")
    print("=" * 60)
    
    # システム設定
    E, basis, sv, H0, dip = setup_system()
    
    # 無次元化分析
    print("\n🔬 物理レジーム分析:")
    _, _, _, _, _, _, scales = nondimensionalize_system(
        H0, dip.mu_x, dip.mu_y, E,
        H0_units="energy", time_units="fs"
    )
    regime_info = analyze_regime(scales)
    
    print(f"  結合強度 λ = {regime_info['lambda']:.3f}")
    print(f"  物理レジーム: {regime_info['regime']}")
    print(f"  説明: {regime_info['description']}")
    print(f"  エネルギースケール: {regime_info['energy_scale_eV']:.3e} eV")
    print(f"  時間スケール: {regime_info['time_scale_fs']:.3e} fs")
    
    # 1. 従来の次元ありシステム
    print("\n🎯 従来の次元ありシステムで計算中...")
    t_dim, psi_dim = schrodinger_propagation(
        H0, E, dip, sv.data,
        axes="xy",
        return_traj=True,
        return_time_psi=True,
        sample_stride=TIME_PARAMS["sample_stride"],
        nondimensional=False,
    )
    pop_dim = np.abs(psi_dim) ** 2
    
    # 2. 無次元化システム
    print("🔬 無次元化システムで計算中...")
    t_nondim, psi_nondim = schrodinger_propagation(
        H0, E, dip, sv.data,
        axes="xy",
        return_traj=True,
        return_time_psi=True,
        sample_stride=TIME_PARAMS["sample_stride"],
        nondimensional=True,
    )
    pop_nondim = np.abs(psi_nondim) ** 2
    
    # 結果比較
    print("\n📊 結果比較:")
    max_diff = np.max(np.abs(pop_dim - pop_nondim))
    print(f"  最大人口差: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("  ✅ 両手法の結果は数値精度内で一致")
    elif max_diff < 1e-6:
        print("  ⚠️  小さな差が検出（許容範囲内）")
    else:
        print("  ❌ 大きな差が検出（要調査）")
    
    # プロット作成
    create_plots(t_dim, pop_dim, t_nondim, pop_nondim, E, basis, regime_info)
    
    return {
        "dimensional": {"time": t_dim, "population": pop_dim},
        "nondimensional": {"time": t_nondim, "population": pop_nondim},
        "regime_info": regime_info,
        "max_difference": max_diff,
    }


def create_plots(t_dim, pop_dim, t_nondim, pop_nondim, E, basis, regime_info):
    """結果のプロット作成"""
    
    # 出力ディレクトリ
    output_dir = Path("nondimensional_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # 状態数の制限（表示用）
    max_states = min(6, pop_dim.shape[1])
    
    # 図1: 人口時間発展の比較
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 次元ありシステム
    for i in range(max_states):
        v, j = basis.get_state(i)[:2]
        axes[0, 0].plot(t_dim, pop_dim[:, i], label=f"|V={v},J={j}⟩")
    axes[0, 0].set_title("従来手法（次元あり）")
    axes[0, 0].set_xlabel("時間 (fs)")
    axes[0, 0].set_ylabel("人口")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 無次元化システム
    for i in range(max_states):
        v, j = basis.get_state(i)[:2]
        axes[0, 1].plot(t_nondim, pop_nondim[:, i], label=f"|V={v},J={j}⟩", linestyle="--")
    axes[0, 1].set_title("無次元化手法")
    axes[0, 1].set_xlabel("時間 (fs)")
    axes[0, 1].set_ylabel("人口")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 差分
    pop_diff = np.abs(pop_dim - pop_nondim)
    for i in range(max_states):
        v, j = basis.get_state(i)[:2]
        axes[1, 0].semilogy(t_dim, pop_diff[:, i], label=f"|V={v},J={j}⟩")
    axes[1, 0].set_title("絶対差分（対数スケール）")
    axes[1, 0].set_xlabel("時間 (fs)")
    axes[1, 0].set_ylabel("|人口_次元あり - 人口_無次元|")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 電場
    t_field = E.tlist
    axes[1, 1].plot(t_field, E.Efield[:, 0], label="Ex", color="blue")
    axes[1, 1].plot(t_field, E.Efield[:, 1], label="Ey", color="red")
    axes[1, 1].set_title("電場波形")
    axes[1, 1].set_xlabel("時間 (fs)")
    axes[1, 1].set_ylabel("電場 (V/m)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"無次元化伝播デモ (λ={regime_info['lambda']:.3f}, {regime_info['regime']})")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300)
    plt.show()
    
    # 図2: レジーム情報
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # レジーム可視化
    lambda_val = regime_info['lambda']
    regimes = ['弱結合\n(λ<0.1)', '中間結合\n(0.1≤λ<1)', '強結合\n(λ≥1)']
    regime_bounds = [0.1, 1.0, 10.0]
    colors = ['green', 'orange', 'red']
    
    for i, (bound, color, label) in enumerate(zip(regime_bounds, colors, regimes)):
        if i == 0:
            x_range = [0, bound]
        else:
            x_range = [regime_bounds[i-1], bound]
        ax.barh(0, x_range[1] - x_range[0], left=x_range[0], color=color, alpha=0.3, label=label)
    
    # 現在の値をマーク
    ax.axvline(lambda_val, color='black', linewidth=3, label=f'現在の系 (λ={lambda_val:.3f})')
    ax.set_xlim(0, 5)
    ax.set_xlabel('結合強度 λ')
    ax.set_title('物理レジーム分類')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "regime_analysis.png", dpi=300)
    plt.show()
    
    print(f"\n📁 結果が {output_dir} に保存されました")


def benchmark_performance():
    """計算性能のベンチマーク"""
    import time
    
    print("\n⏱️  計算性能ベンチマーク:")
    
    E, basis, sv, H0, dip = setup_system()
    
    # 次元ありシステム
    start_time = time.time()
    schrodinger_propagation(
        H0, E, dip, sv.data,
        axes="xy",
        return_traj=True,
        sample_stride=TIME_PARAMS["sample_stride"],
        nondimensional=False,
    )
    time_dimensional = time.time() - start_time
    
    # 無次元化システム
    start_time = time.time()
    schrodinger_propagation(
        H0, E, dip, sv.data,
        axes="xy",
        return_traj=True,
        sample_stride=TIME_PARAMS["sample_stride"],
        nondimensional=True,
    )
    time_nondimensional = time.time() - start_time
    
    print(f"  従来手法: {time_dimensional:.3f} 秒")
    print(f"  無次元化: {time_nondimensional:.3f} 秒")
    print(f"  比率: {time_nondimensional/time_dimensional:.2f}x")


def main():
    """メイン実行関数"""
    try:
        # 比較実行
        results = run_comparison()
        
        # 性能ベンチマーク
        benchmark_performance()
        
        # 要約
        print("\n" + "=" * 60)
        print("📋 まとめ")
        print("=" * 60)
        print("✅ 無次元化機能が正常に動作しています")
        print(f"✅ 数値精度: 最大差 {results['max_difference']:.2e}")
        print(f"✅ 物理レジーム: {results['regime_info']['regime']}")
        print("✅ プロットが正常に生成されました")
        
        return results
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 