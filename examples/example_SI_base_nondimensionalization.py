"""
SI基本単位ベースの無次元化システムデモ
=====================================

このサンプルは、デフォルト単位を維持しながら、無次元化の際に
SI基本単位（接頭辞なし）に自動変換する新しいシステムのデモです。

主な特徴:
- デフォルト単位設定は変更せず、cm⁻¹、D、MV/cm、eV、fsを維持
- 無次元化の際に自動的にSI基本単位（rad/s、C·m、V/m、J、s）に変換
- 物理的に一貫したスケーリング

実行方法:
python examples/example_SI_base_nondimensionalization.py
"""

import numpy as np
from rovibrational_excitation.core.nondimensionalize import (
    create_SI_demo_parameters,
    convert_default_units_to_SI_base,
    determine_SI_based_scales,
    nondimensionalize_with_SI_base_units,
    analyze_regime,
)
from rovibrational_excitation.core.units import (
    get_default_units,
    auto_convert_parameters,
)


def demo_default_units_preservation():
    """デフォルト単位設定の維持を確認"""
    print("=" * 60)
    print("📋 DEFAULT UNITS PRESERVATION DEMO")
    print("=" * 60)
    
    defaults = get_default_units()
    print("\n🎯 Current default units (unchanged):")
    for quantity, unit in defaults.items():
        print(f"   {quantity}: {unit}")
    
    print("\n💡 These default units are preserved while enabling SI-based nondimensionalization!")


def demo_SI_base_conversion():
    """SI基本単位変換デモ"""
    print("\n" + "=" * 60)
    print("🔄 SI BASE UNIT CONVERSION DEMO")
    print("=" * 60)
    
    # デフォルト単位でのサンプル値
    print("\n📊 Sample values in default units:")
    freq_cm_inv = 2349.1  # cm⁻¹
    dipole_D = 0.3        # D
    field_MV_cm = 5.0     # MV/cm
    energy_eV = 1.5       # eV
    time_fs = 30.0        # fs
    
    print(f"   Frequency: {freq_cm_inv} cm⁻¹")
    print(f"   Dipole: {dipole_D} D")
    print(f"   Field: {field_MV_cm} MV/cm")
    print(f"   Energy: {energy_eV} eV")
    print(f"   Time: {time_fs} fs")
    
    # SI基本単位への変換
    print("\n🔄 Converting to SI base units...")
    freq_rad_s, dipole_Cm, field_V_m, energy_J, time_s = convert_default_units_to_SI_base(
        freq_cm_inv, dipole_D, field_MV_cm, energy_eV, time_fs
    )
    
    print(f"\n📐 Physical validation:")
    print(f"   Frequency corresponds to wavelength: {2*np.pi*299792458/freq_rad_s*1e6:.1f} μm")
    print(f"   Energy in thermal units: {energy_J/(1.380649e-23*300):.1f} kT (at 300K)")
    print(f"   Time in atomic units: {time_s*4.134e16:.1f} a.u.")


def demo_complete_nondimensionalization():
    """完全な無次元化ワークフローデモ"""
    print("\n" + "=" * 60)
    print("🔢 COMPLETE NONDIMENSIONALIZATION DEMO")
    print("=" * 60)
    
    # 様々な単位系のパラメータを作成
    mixed_params = {
        "omega_rad_phz": 70.4,          # THz (異なる周波数単位)
        "omega_rad_phz_units": "THz",
        
        "mu0_Cm": 1.0,                  # 原子単位
        "mu0_Cm_units": "ea0",
        
        "amplitude": 100,               # kV/cm (異なる電場単位)
        "amplitude_units": "kV/cm",
        
        "energy_gap": 300,              # meV (異なるエネルギー単位)
        "energy_gap_units": "meV",
        
        "duration": 0.1,                # ps (異なる時間単位)
        "duration_units": "ps",
    }
    
    print("\n🔧 Input parameters (mixed units):")
    for k, v in mixed_params.items():
        if not k.endswith("_units"):
            unit_key = f"{k}_units"
            unit = mixed_params.get(unit_key, "")
            print(f"   {k}: {v} {unit}")
    
    # 自動変換（デフォルト単位経由でSI単位へ）
    print("\n🎯 Converting via default units to SI...")
    converted = auto_convert_parameters(mixed_params)
    
    # 簡単な2準位系を作成
    energy_gap_J = converted["energy_gap"]
    H0 = np.array([0.0, energy_gap_J])
    
    mu_value = converted["mu0_Cm"]
    mu_x = np.array([[0.0, mu_value], [mu_value, 0.0]])
    mu_y = np.zeros_like(mu_x)
    
    amplitude = converted["amplitude"]
    
    print(f"\n📊 Quantum system in SI base units:")
    print(f"   H0 = diag([0, {energy_gap_J:.6e}]) J")
    print(f"   μ_max = {mu_value:.6e} C·m")
    print(f"   E_field = {amplitude:.6e} V/m")
    
    # SI基本単位ベースのスケール決定
    print("\n📏 Determining nondimensionalization scales...")
    scales = determine_SI_based_scales(H0, mu_x, amplitude)
    
    # 無次元化の実行
    H0_prime = H0 / scales.E0
    mu_x_prime = mu_x / scales.mu0
    
    print(f"\n✨ Nondimensionalized results:")
    print(f"   H0' = {H0_prime}")
    print(f"   μ_x' max = {np.max(np.abs(mu_x_prime)):.3f}")
    print(f"   E' = 1.0 (normalized)")
    
    # 物理レジームの分析
    regime_info = analyze_regime(scales)
    print(f"\n🎭 Physical regime analysis:")
    print(f"   Regime: {regime_info['regime']}")
    print(f"   Description: {regime_info['description']}")
    print(f"   λ = {regime_info['lambda']:.3f}")


def demo_different_scale_regimes():
    """異なる物理スケールレジームのデモ"""
    print("\n" + "=" * 60)
    print("⚖️ DIFFERENT PHYSICAL SCALE REGIMES DEMO")
    print("=" * 60)
    
    regimes = [
        {
            "name": "Molecular vibrations",
            "energy_eV": 0.1,  # IR region
            "dipole_D": 0.1,   # small dipole
            "field_MV_cm": 0.1,  # weak field
        },
        {
            "name": "Electronic transitions",
            "energy_eV": 2.0,   # visible region
            "dipole_D": 1.0,    # strong dipole
            "field_MV_cm": 10.0, # strong field
        },
        {
            "name": "X-ray transitions",
            "energy_eV": 1000.0, # keV region
            "dipole_D": 0.01,   # weak dipole
            "field_MV_cm": 100.0, # very strong field
        },
    ]
    
    for i, regime in enumerate(regimes, 1):
        print(f"\n{i}. {regime['name']}:")
        print(f"   Input: {regime['energy_eV']} eV, {regime['dipole_D']} D, {regime['field_MV_cm']} MV/cm")
        
        # Convert to SI
        _, dipole_Cm, field_V_m, energy_J, _ = convert_default_units_to_SI_base(
            1000, regime["dipole_D"], regime["field_MV_cm"], regime["energy_eV"], 10.0
        )
        
        # Create system
        H0 = np.array([0.0, energy_J])
        mu_x = np.array([[0.0, dipole_Cm], [dipole_Cm, 0.0]])
        
        # Determine scales
        scales = determine_SI_based_scales(H0, mu_x, field_V_m)
        
        # Analyze regime
        regime_info = analyze_regime(scales)
        print(f"   Result: {regime_info['regime']} (λ = {regime_info['lambda']:.3f})")


def main():
    """メイン実行関数"""
    print("🚀 SI BASE UNIT NONDIMENSIONALIZATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows the new nondimensionalization system that:")
    print("• Preserves default unit settings (cm⁻¹, D, MV/cm, eV, fs)")
    print("• Automatically converts to SI base units for nondimensionalization")
    print("• Ensures physical consistency and proper scaling")
    
    try:
        demo_default_units_preservation()
        demo_SI_base_conversion()
        demo_complete_nondimensionalization()
        demo_different_scale_regimes()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n💡 Key achievements:")
        print("   • Default units preserved for user convenience")
        print("   • Automatic SI base unit conversion for nondimensionalization")
        print("   • Physics-based scaling with proper regime identification")
        print("   • Seamless integration with existing unit system")
        print("   • Support for mixed input units")
        
        print("\n🎯 Usage recommendation:")
        print("   Use nondimensionalize_with_SI_base_units() for all")
        print("   nondimensionalization needs with automatic unit handling!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 