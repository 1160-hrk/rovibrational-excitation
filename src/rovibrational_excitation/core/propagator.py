"""
rovibrational_excitation/core/propagator.py
------------------------------------------
* axes="xy"  → Ex ↔ μ_x,  Ey ↔ μ_y   (デフォルト)
* axes="zx"  → Ex ↔ μ_z,  Ey ↔ μ_x
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

# ---------------------------------------------------------------------
# RK4 kernels
from ._rk4_lvne import rk4_lvne, rk4_lvne_traj
from ._rk4_schrodinger import rk4_schrodinger
from ._splitop_schrodinger import splitop_schrodinger
from .nondimensionalize import nondimensionalize_system, get_physical_time

# ---------------------------------------------------------------------
# optional CuPy
try:
    import cupy as _cp  # noqa: N811
except ImportError:
    _cp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------
# type-hints
if TYPE_CHECKING:
    Array = Union[np.ndarray, "_cp.ndarray"]
    from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix

    from .electric_field import ElectricField
else:
    Array = np.ndarray  # runtime dummy

# ---------------------------------------------------------------------
# constants / helpers
_DIRAC_HBAR = 6.62607015e-019 / (2 * np.pi)  # J fs


def validate_propagation_units(
    H0, 
    dipole_matrix, 
    efield, 
    expected_H0_units="J", 
    expected_dipole_units="C*m"
):
    """
    Validate units before propagation and warn about potential issues.
    
    Parameters
    ----------
    H0 : Array
        Hamiltonian matrix
    dipole_matrix : LinMolDipoleMatrix
        Dipole moment matrices
    efield : ElectricField
        Electric field object
    expected_H0_units : str, default "J"
        Expected units for Hamiltonian ("J", "eV", "rad/fs", "cm^-1")
    expected_dipole_units : str, default "C*m"
        Expected units for dipole moments ("C*m", "D", "ea0")
        
    Returns
    -------
    list
        List of warning messages (empty if no issues)
        
    Examples
    --------
    >>> warnings = validate_propagation_units(H0, dipole_matrix, efield, "eV", "D")
    >>> if warnings:
    ...     for w in warnings:
    ...         print(f"⚠️  {w}")
    """
    warnings = []
    
    try:
        # Energy scale analysis
        if H0.ndim == 2:
            eigenvals = np.diag(H0)
        else:
            eigenvals = H0
        
        energy_range = np.ptp(eigenvals)  # peak-to-peak range
        max_energy = np.max(np.abs(eigenvals))
        
        # Unit-specific sanity checks
        if expected_H0_units == "J":
            # J units: typical molecular range 1e-25 to 1e-15 J
            if energy_range < 1e-25 or energy_range > 1e-15:
                warnings.append(
                    f"エネルギー範囲 {energy_range:.2e} J が分子系として異常です "
                    f"(期待範囲: 1e-25 - 1e-15 J)"
                )
        elif expected_H0_units == "eV":
            # eV units: typical range 1e-6 to 100 eV
            if energy_range < 1e-6 or energy_range > 100:
                warnings.append(
                    f"エネルギー範囲 {energy_range:.2e} eV が異常です "
                    f"(期待範囲: 1e-6 - 100 eV)"
                )
        elif expected_H0_units == "cm^-1":
            # cm⁻¹ units: typical range 0.1 to 10000 cm⁻¹
            if energy_range < 0.1 or energy_range > 10000:
                warnings.append(
                    f"エネルギー範囲 {energy_range:.2e} cm⁻¹ が異常です "
                    f"(期待範囲: 0.1 - 10000 cm⁻¹)"
                )
        elif expected_H0_units == "rad/fs":
            # rad/fs units: converted from typical molecular frequencies
            if energy_range < 1e-6 or energy_range > 1e3:
                warnings.append(
                    f"周波数範囲 {energy_range:.2e} rad/fs が異常です "
                    f"(期待範囲: 1e-6 - 1e3 rad/fs)"
                )
        
        # Dipole moment analysis
        mu_x = _pick_mu(dipole_matrix, 'x')
        mu_y = _pick_mu(dipole_matrix, 'y')
        max_dipole = max(np.max(np.abs(mu_x)), np.max(np.abs(mu_y)))
        
        if expected_dipole_units == "C*m":
            # C·m units: typical range 1e-35 to 1e-25 C·m
            if max_dipole < 1e-35 or max_dipole > 1e-25:
                warnings.append(
                    f"双極子モーメント {max_dipole:.2e} C·m が異常です "
                    f"(期待範囲: 1e-35 - 1e-25 C·m)"
                )
        elif expected_dipole_units == "D":
            # Debye units: typical range 0.001 to 100 D
            if max_dipole < 0.001 or max_dipole > 100:
                warnings.append(
                    f"双極子モーメント {max_dipole:.2e} D が異常です "
                    f"(期待範囲: 0.001 - 100 D)"
                )
        elif expected_dipole_units == "ea0":
            # Atomic units: typical range 0.01 to 10 ea0
            if max_dipole < 0.01 or max_dipole > 10:
                warnings.append(
                    f"双極子モーメント {max_dipole:.2e} ea0 が異常です "
                    f"(期待範囲: 0.01 - 10 ea0)"
                )
        
        # Time scale analysis (if energy range is reasonable)
        if energy_range > 0:
            # Characteristic time: τ = ℏ/ΔE
            if expected_H0_units == "J":
                char_time_fs = _DIRAC_HBAR / energy_range * 1e15
            elif expected_H0_units == "rad/fs":
                char_time_fs = 1 / energy_range  # already in fs⁻¹
            else:
                # For other units, convert to estimate
                char_time_fs = 1000  # rough estimate
            
            if hasattr(efield, 'dt') and efield.dt > char_time_fs / 5:
                warnings.append(
                    f"時間ステップ {efield.dt:.3f} fs が特性時間 "
                    f"{char_time_fs:.3f} fs に対して大きすぎます "
                    f"(推奨: < {char_time_fs/5:.3f} fs)"
                )
        
        # Electric field magnitude check
        if hasattr(efield, 'Efield'):
            max_field = np.max(np.abs(efield.Efield))
            if max_field > 1e12:  # > 1 TV/m is extreme
                warnings.append(
                    f"電場強度 {max_field:.2e} V/m が極端に大きいです"
                )
            elif max_field < 1e3:  # < 1 kV/m is very weak
                warnings.append(
                    f"電場強度 {max_field:.2e} V/m が非常に弱いです"
                )
            
            # Interaction strength analysis
            if max_dipole > 0 and energy_range > 0:
                # Rough interaction strength: μE/ΔE
                if expected_H0_units == "J" and expected_dipole_units == "C*m":
                    interaction_strength = max_field * max_dipole / energy_range
                    if interaction_strength > 0.1:
                        warnings.append(
                            f"強電場域です (相互作用強度 = {interaction_strength:.3f}). "
                            "小さな時間ステップを検討してください"
                        )
        
    except Exception as e:
        warnings.append(f"単位検証中にエラーが発生しました: {e}")
    
    return warnings


def _cm_to_rad_phz(mu: Array) -> Array:
    """μ (C·m) → rad / (PHz/(V·m⁻¹))."""
    return mu / (_DIRAC_HBAR)  # divide once; xp 対応は呼び出し側で

def _J_to_rad_phz(H0: Array) -> Array:
    """J → rad / fs."""
    return H0 / _DIRAC_HBAR

def _backend(name: str):
    if name == "cupy":
        if _cp is None:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
        return _cp
    return np


def _pick_mu(dip, axis: str) -> Array:
    """
    Get dipole matrix in SI units (C·m) for specified axis.
    
    Parameters
    ----------
    dip : LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
        Dipole matrix object with unit management
    axis : str
        Dipole axis ('x', 'y', 'z')
        
    Returns
    -------
    Array
        Dipole matrix in SI units (C·m)
    """
    # Try to get SI units first (preferred method)
    si_method = f"get_mu_{axis}_SI"
    if hasattr(dip, si_method):
        return getattr(dip, si_method)()
    
    # Fallback to direct attribute access (for backward compatibility)
    attr = f"mu_{axis}"
    if not hasattr(dip, attr):
        raise AttributeError(f"{type(dip).__name__} has no attribute '{attr}' or '{si_method}'")
    return getattr(dip, attr)


# ---------------------------------------------------------------------
def _prepare_args(
    H0: Array,
    E: ElectricField,
    dip,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    *,
    axes: str = "xy",
    dt: float | None = None,
    mu_x_override: Array | None = None,
    mu_y_override: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array, float, int]:
    """
    共通前処理

    Returns (順序は旧バージョンと互換):
        H0, μ_a, μ_b, Ex, Ey, dt, steps
        └─ μ_a: Ex に対応 / μ_b: Ey に対応
    """
    axes = axes.lower()
    if len(axes) != 2 or any(a not in "xyz" for a in axes):
        raise ValueError("axes must be like 'xy', 'zx', ...")

    ax0, ax1 = axes[0], axes[1]
    xp = _cp if _cp is not None else np

    dt_half = E.dt if dt is None else dt / 2

    Ex, Ey = E.Efield[:, 0], E.Efield[:, 1]

    # stepsを計算
    steps = (len(Ex) - 1) // 2

    # スパース行列対応: スパース行列の場合はそのまま使用
    mu_a_raw = _pick_mu(dip, ax0)
    mu_b_raw = _pick_mu(dip, ax1)

    try:
        import scipy.sparse as sp

        if sp.issparse(mu_a_raw):
            mu_a = _cm_to_rad_phz(mu_a_raw)  # スパース行列の場合はそのまま
        else:
            mu_a = xp.asarray(_cm_to_rad_phz(mu_a_raw))
        if sp.issparse(mu_b_raw):
            mu_b = _cm_to_rad_phz(mu_b_raw)  # スパース行列の場合はそのまま
        else:
            mu_b = xp.asarray(_cm_to_rad_phz(mu_b_raw))
    except ImportError:
        mu_a = xp.asarray(_cm_to_rad_phz(mu_a_raw))
        mu_b = xp.asarray(_cm_to_rad_phz(mu_b_raw))

    return (
        xp.asarray(_J_to_rad_phz(H0)),
        mu_a,
        mu_b,
        xp.asarray(Ex),
        xp.asarray(Ey),
        dt_half * 2,
        steps,
    )


# ---------------------------------------------------------------------
def schrodinger_propagation(
    hamiltonian,  # Type: Hamiltonian
    Efield: ElectricField,
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    psi0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    return_time_psi: bool = False,
    sample_stride: int = 1,
    backend: str = "numpy",
    sparse: bool = False,
    nondimensional: bool = False,
    validate_units: bool = True,
    verbose: bool = False,
    renorm: bool = False,
) -> Array:
    """
    Time-dependent Schrödinger equation propagator with unit-aware physics objects.
    
    Enhanced version that uses Hamiltonian and LinMolDipoleMatrix objects with
    internal unit management for robust physics calculations.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object with internal unit management
    Efield : ElectricField
        Electric field object  
    dipole_matrix : LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
        Dipole moment matrices with internal unit management
    psi0 : Array
        Initial wavefunction
    axes : str, default "xy"
        Polarization axes mapping ("xy", "zx", etc.)
    return_traj : bool, default True
        Return full trajectory vs final state only
    return_time_psi : bool, default False
        Return time array along with trajectory
    sample_stride : int, default 1
        Sampling stride for trajectory
    backend : str, default "numpy" 
        Computational backend ("numpy" or "cupy")
    nondimensional : bool, default False
        Use nondimensional propagation
    validate_units : bool, default True
        Perform unit validation before propagation
        
    Returns
    -------
    Array or tuple
        Propagated wavefunction(s), optionally with time array
        
    Notes
    -----
    This function now uses unit-aware physics objects that handle unit
    conversion internally. All physics quantities automatically use
    SI units (J, C·m, V/m) for calculation consistency.
    
    Examples
    --------
    >>> # Using Hamiltonian object
    >>> h = Hamiltonian(H0_matrix, "eV")  # Created in eV
    >>> dipole = LinMolDipoleMatrix(basis, mu0=0.3, units="D")  # In Debye
    >>> result = schrodinger_propagation(h, efield, dipole, psi0)
    
    >>> # Objects handle unit conversion automatically
    >>> h_matrix_J = h.get_matrix("J")  # Automatic eV → J conversion
    >>> mu_x_SI = dipole.get_mu_x_SI()  # Automatic D → C·m conversion
    """
    
    # Extract quantities in SI units from unit-aware objects
    H0_SI = hamiltonian.get_matrix("J")  # Always get in J (SI energy units)
    mu_x_SI = dipole_matrix.get_mu_x_SI()  # Always get in C·m (SI dipole units)
    mu_y_SI = dipole_matrix.get_mu_y_SI()
    
    # Get electric field in SI units (V/m) and time in fs
    efield_SI = Efield.get_Efield_SI()  # Always get in V/m
    time_fs = Efield.get_time_SI()  # Always get in fs
    
    # Unit validation using SI quantities
    if validate_units:
        # Create a temporary dipole matrix-like object for validation
        class TempDipole:
            def __init__(self, mu_x, mu_y):
                self.mu_x = mu_x
                self.mu_y = mu_y
        
        temp_dipole = TempDipole(mu_x_SI, mu_y_SI)
        warnings = validate_propagation_units(
            H0_SI, temp_dipole, Efield, "J", "C*m"
        )
        if warnings:
            print("⚠️  単位検証で以下の警告が検出されました:")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
            
            # For non-interactive environments, don't halt execution
            # but make warnings prominent
            if len(warnings) >= 3:
                print("\n🚨 複数の重大な警告があります。計算結果を慎重にご確認ください。")
        else:
            print("✅ 単位検証: すべて正常です")
    
    # Get field scale information
    field_scale_info = Efield.get_field_scale_info()
    
    # Display unit information for transparency
    if verbose:
        print(f"🔧 Using physics objects with internal unit management:")
        print(f"   Hamiltonian: {hamiltonian.units} → J (automatic conversion)")
        print(f"   Dipole matrix: {dipole_matrix.units} → C·m (automatic conversion)")
        print(f"   Electric field: {Efield.field_units} → V/m, time: {Efield.time_units} → fs")
        print(f"   Field scale (auto-determined): {field_scale_info['scale_V_per_m']:.2e} V/m")
        print(f"     = {field_scale_info['scale_MV_per_cm']:.2f} MV/cm")
        print(f"     = {field_scale_info['scale_in_original_units']:.2f} {field_scale_info['original_units']}")
    
    # Use SI quantities for all calculations
    H0_converted = H0_SI

    # backend引数の型チェック
    if backend not in ("numpy", "cupy"):
        raise ValueError("backend must be 'numpy' or 'cupy'")

    backend_typed: Literal["numpy", "cupy"] = backend  # type: ignore[assignment]

    # 無次元化処理
    if nondimensional:
        # SI単位で双極子行列を取得
        mu_x_raw = mu_x_SI
        mu_y_raw = mu_y_SI
        
        # dtを_prepare_argsと同じ方法で調整
        dt_for_nondim = Efield.dt * 2  # _prepare_argsと同じ: dt_half * 2
        
        # 無次元化実行
        (
            H0_prime,
            mu_x_prime,
            mu_y_prime,
            Efield_prime,
            tlist_prime,
            dt_prime,
            scales,
        ) = nondimensionalize_system(
            H0_SI, mu_x_raw, mu_y_raw, Efield,
            H0_units="energy",  # H0はエネルギー単位（J）
            time_units="fs",    # 時間はfs単位
            dt=dt_for_nondim,   # 調整されたdtを使用
        )
        
        # 無次元化されたパラメータで時間発展実行
        xp = _backend(backend)
        
        # 無次元電場を準備
        Ex_prime = Efield_prime[:, 0]
        Ey_prime = Efield_prime[:, 1]
        steps_prime = (len(Ex_prime) - 1) // 2
        
        # 無次元化では結合強度λがかかるため調整
        mu_x_eff = xp.asarray(mu_x_prime * scales.lambda_coupling)
        mu_y_eff = xp.asarray(mu_y_prime * scales.lambda_coupling)
        
        # ---------------------------------------------------------
        # 無次元化システムでもsplit-operator法を試す
        # ---------------------------------------------------------
        try:
            # 元のElectricFieldから直接偏光情報を取得して無次元化
            if (isinstance(Efield._constant_pol, np.ndarray) 
                and hasattr(Efield, '_scalar_field') 
                and Efield._scalar_field is not None):
                
                # 偏光ベクトルはそのまま（無次元）
                pol = Efield._constant_pol
                
                # スカラー場を無次元化
                Escalar_prime = Efield._scalar_field / scales.Efield0
            else:
                # 偏光が時間依存の場合はValueErrorを発生させてRK4にフォールバック
                raise ValueError("Polarization is time-dependent")
            
            traj_split = splitop_schrodinger(
                xp.asarray(H0_prime),
                mu_x_eff,
                mu_y_eff,
                pol,
                xp.asarray(Escalar_prime),
                xp.asarray(psi0),
                dt_prime,
                steps=steps_prime,
                sample_stride=sample_stride,
                hbar=1.0,  # 無次元系ではhbar=1
                backend=backend_typed,
            )
            
            # 形状を調整
            result = traj_split.squeeze()
            if result.ndim == 1:
                result = result.reshape(1, -1)

            if return_traj:
                if return_time_psi:
                    time_psi = get_physical_time(
                        xp.arange(0, result.shape[0]) * dt_prime * sample_stride, scales
                    )
                    return time_psi, result
                else:
                    return result
            else:
                return result[-1:].reshape((1, len(psi0)))
                
        except (ValueError, AttributeError):
            # 偏光が時間依存、またはElectricFieldインポートエラー → RK4へフォールバック
            pass
        
        # RK4実行（フォールバック）
        result = rk4_schrodinger(
            xp.asarray(H0_prime),
            mu_x_eff,
            mu_y_eff,
            xp.asarray(Ex_prime),
            xp.asarray(Ey_prime),
            xp.asarray(psi0),
            dt_prime,
            sample_stride,
            renorm,
            sparse,
            backend=backend_typed,
        )
        
        if return_traj:
            if return_time_psi:
                time_psi = get_physical_time(
                    xp.arange(0, result.shape[0]) * dt_prime * sample_stride, scales
                )
                return time_psi, result
            else:
                return result
        else:
            return result[-1:].reshape((1, len(psi0)))
    
    # 従来の次元ありシステム
    xp = _backend(backend)
    H0_, mu_a, mu_b, Ex, Ey, dt, steps = _prepare_args(
        H0_converted, Efield, dipole_matrix, axes=axes
    )
    # ---------------------------------------------------------
    # 0) まず split-operator が適用できるか試す
    #    （ElectricField に「一定偏光＋実スカラー場」が
    #      保持されている場合だけ使用）
    # ---------------------------------------------------------
    try:
        Escalar, pol = Efield.get_scalar_and_pol()  # ← ElectricField で追加した util

        traj_split = splitop_schrodinger(
            H0_,
            mu_a,
            mu_b,  # μ_x, μ_y
            pol,  # (2,) complex
            Escalar,  # (N,) real
            xp.asarray(psi0),
            dt,
            steps=(len(Escalar) - 1) // 2,
            sample_stride=sample_stride,
            backend=backend_typed,
        )

        # 形状を調整
        result = traj_split.squeeze()
        if result.ndim == 1:
            result = result.reshape(1, -1)

        if return_traj:
            if return_time_psi:
                # resultがtupleの場合はshapeアクセスできないので修正
                if isinstance(result, tuple):
                    # すでにtupleになっている場合はそのまま返す
                    return result
                time_psi = xp.arange(
                    0, result.shape[0] * dt * sample_stride, dt * sample_stride
                )
                return time_psi, result
            return result
        else:
            return result[-1:].reshape((1, len(psi0)))

    except ValueError:
        # 偏光が時間依存 → 旧来の RK4 へフォールバック
        pass
    result = rk4_schrodinger(
        H0_, mu_a, mu_b, Ex, Ey, xp.asarray(psi0), dt, sample_stride, renorm, sparse, backend=backend_typed)

    if return_traj:
        if return_time_psi:
            time_psi = xp.arange(0, result.shape[0] * dt * sample_stride, dt * sample_stride)
            return time_psi, result
        else:
            return result
    else:
        return result[-1:].reshape((1, len(psi0)))



# ---------------------------------------------------------------------
def mixed_state_propagation(
    H0: Array,
    Efield: ElectricField,
    psi0_array: Iterable[Array],
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    *,
    axes: str = "xy",
    return_traj: bool = True,
    return_time_rho: bool = False,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)

    # Iterableをリストに変換してインデックスアクセス可能にする
    psi0_list = list(psi0_array)
    dim = psi0_list[0].shape[0]

    steps_out = (len(Efield.tlist) // 2) // sample_stride + 1
    rho_out = (
        xp.zeros((steps_out, dim, dim), dtype=xp.complex128)
        if return_traj
        else xp.zeros((dim, dim), dtype=xp.complex128)
    )

    for psi0 in psi0_list:
        result = schrodinger_propagation(
            H0,
            Efield,
            dipole_matrix,
            psi0,
            axes=axes,
            return_traj=return_traj,
            return_time_psi=False,  # time情報は不要
            sample_stride=sample_stride,
            backend=backend,
        )

        # resultがtupleの場合の処理
        if isinstance(result, tuple):
            psi_t = result[1]
        else:
            psi_t = result

        if return_traj:
            rho_out += xp.einsum("ti, tj -> tij", psi_t, psi_t.conj())
        else:
            rho_out += psi_t[0] @ psi_t[0].conj().T

    if return_traj:
        if return_time_rho:
            dt_rho = Efield.dt_state * sample_stride
            steps = Efield.steps_state
            time_psi = xp.arange(0, (steps + 1) * dt_rho, dt_rho)
            return time_psi, rho_out
        else:
            return rho_out
    else:
        return rho_out


# ---------------------------------------------------------------------
def liouville_propagation(
    H0: Array,
    Efield: ElectricField,
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    rho0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    H0_, mu_a, mu_b, Ex, Ey, dt, steps = _prepare_args(
        H0, Efield, dipole_matrix, axes=axes
    )

    # 引数の数を修正 - rk4_lvne_trajとrk4_lvneで異なる引数を正しく渡す
    rk4_args = (H0_, mu_a, mu_b, Ex, Ey, xp.asarray(rho0), dt, steps)

    if return_traj:
        return rk4_lvne_traj(*rk4_args, sample_stride)
    else:
        return rk4_lvne(*rk4_args)
