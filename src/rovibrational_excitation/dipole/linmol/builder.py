"""
rovibrational_excitation.dipole.linmol/builder.py
========================
μ-axis 行列 (x, y, z) を高速生成

* potential_type = 'harmonic' | 'morse'
* CPU : Numba  (dense／CSR)
* GPU : CuPy   (broadcast → dense／CSR)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import numpy as _np
import scipy.sparse as _sp
from numba import njit, prange

try:  # GPU (optional)
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
except ImportError:
    _cp = None
    _csp = None

from rovibrational_excitation.dipole.rot.jm import tdm_jm_x, tdm_jm_y, tdm_jm_z
from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm  # Python 版
from rovibrational_excitation.dipole.vib.morse import (
    omega01_domega_to_N,
    tdm_vib_morse,
)  # Python 版

# ----------------------------------------------------------------------
# 型エイリアス
# ----------------------------------------------------------------------
if TYPE_CHECKING:
    from rovibrational_excitation.core.basis import LinMolBasis

if _cp is not None:
    Array: type = Union[_np.ndarray, _cp.ndarray]  # type: ignore
else:
    Array: type = _np.ndarray  # type: ignore


# ----------------------------------------------------------------------
# --- 1. 小さな "ラッパー" を JIT しておく ------------------------------
#   * Numba カーネル側で未型 global が残らないようにする
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def _vib_harm(v1: int, v2: int) -> float:
    """√v 選択則 (Δv = ±1)"""
    if v1 == v2 + 1:
        return _np.sqrt(v1)
    elif v2 == v1 + 1:
        return _np.sqrt(v2)
    else:
        return 0.0


_vib_morse_jit = njit(cache=True, fastmath=True)(tdm_vib_morse)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# --- 2. 共通 Numba カーネル (dense) -----------------------------------
#   * axis_idx : 0→x, 1→y, 2→z
#   * vib_is_morse : True なら Morse、False なら Harmonic
# ----------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _dense_core(v_arr, J_arr, M_arr, mu0: float, axis_idx: int, vib_is_morse: bool):
    dim = v_arr.size
    out = _np.zeros((dim, dim), dtype=_np.complex128)

    for i in prange(dim):
        v1, J1, M1 = v_arr[i], J_arr[i], M_arr[i]
        # if J1 == 0:
        #     continue
        for j in range(dim):
            v2, J2, M2 = v_arr[j], J_arr[j], M_arr[j]
            # if J2 == 0:
            #     continue

            # --- 選択則 ------------------------------------------------
            # if abs(v1 - v2) != 1:
            #     continue
            if abs(J1 - J2) != 1:
                continue
            if abs(J1 - J2) > 1:  # 回転選択則: ΔJ = 0, ±1
                continue
            if abs(M1 - M2) > 1:  # 磁気選択則: ΔM = 0, ±1
                continue

            # --- 回転要素 ----------------------------------------------
            if axis_idx == 0:
                r = tdm_jm_x(J1, M1, J2, M2)
            elif axis_idx == 1:
                r = tdm_jm_y(J1, M1, J2, M2)
            else:
                r = tdm_jm_z(J1, M1, J2, M2)
            if r == 0.0:
                continue

            # --- 振動ファクター ----------------------------------------
            vfac = _vib_morse_jit(v1, v2) if vib_is_morse else _vib_harm(v1, v2)
            if vfac == 0.0:
                continue

            out[i, j] = mu0 * r * vfac
    return out


# ----------------------------------------------------------------------
# --- 3. CPU sparse (Python loop) --------------------------------------
# ----------------------------------------------------------------------
def _sparse_cpu(v_arr, J_arr, M_arr, mu0: float, axis_idx: int, vib_func):
    rot_func = (tdm_jm_x, tdm_jm_y, tdm_jm_z)[axis_idx]
    data, row, col = [], [], []

    for i, (v1, J1, M1) in enumerate(zip(v_arr, J_arr, M_arr)):
        # J=0状態もスキップしない（dense版と同じ）
        mask = (abs(J1 - J_arr) <= 1) & (  # 回転選択則: ΔJ = 0, ±1
            abs(M1 - M_arr) <= 1
        )  # 磁気選択則: ΔM = 0, ±1
        for j in _np.nonzero(mask)[0]:
            r = rot_func(J1, M1, J_arr[j], M_arr[j])
            if r == 0.0:
                continue
            vfac = vib_func(v1, v_arr[j])
            if vfac == 0.0:
                continue
            data.append(mu0 * r * vfac)
            row.append(i)
            col.append(j)

    # 空の場合はshapeを明示的に指定
    shape = (len(v_arr), len(v_arr))
    return _sp.csr_matrix((data, (row, col)), shape=shape, dtype=_np.complex128)


# ----------------------------------------------------------------------
# --- 4. GPU dense helper ---------------------------------------------
#   CuPy では Python vectorize を使うのでラップ不要
# ----------------------------------------------------------------------
def _dense_gpu(v_arr, J_arr, M_arr, mu0: float, axis_idx: int, vib_func):
    xp = _cp
    rot_func = (tdm_jm_x, tdm_jm_y, tdm_jm_z)[axis_idx]

    v1 = xp.expand_dims(v_arr, 1)
    v2 = xp.expand_dims(v_arr, 0)
    J1 = xp.expand_dims(J_arr, 1)
    J2 = xp.expand_dims(J_arr, 0)
    M1 = xp.expand_dims(M_arr, 1)
    M2 = xp.expand_dims(M_arr, 0)

    mask = (
        (xp.abs(v1 - v2) == 1)
        & (xp.abs(J1 - J2) == 1)
        & (xp.abs(M1 - M2) <= 1)
        & (J1 != 0)
        & (J2 != 0)
    )

    rot = xp.vectorize(rot_func, otypes=[xp.complex128])(J1, M1, J2, M2)
    vib = xp.vectorize(vib_func, otypes=[xp.float64])(v1, v2)
    return mu0 * rot * vib * mask


# ----------------------------------------------------------------------
# --- 5. Public API ----------------------------------------------------
# ----------------------------------------------------------------------
def build_mu(
    basis: LinMolBasis,
    axis: Literal["x", "y", "z"],
    mu0: float,
    *,
    potential_type: Literal["harmonic", "morse"] = "harmonic",
    backend: Literal["numpy", "cupy"] = "numpy",
    dense: bool = True,
):
    """
    Generate transition-dipole matrix μ_axis.

    Parameters
    ----------
    basis : LinMolBasis   (must expose .V_array, .J_array, and optionally .M_array)
    axis  : 'x' | 'y' | 'z'
    mu0   : overall scaling
    potential_type : 'harmonic' | 'morse'
    backend : 'numpy' | 'cupy'
    dense  : True → ndarray, False → CSR
    """
    # 大文字小文字を区別しないように正規化してからチェック
    axis_normalized = axis.lower()
    if axis_normalized not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    pot = potential_type.lower()
    if pot not in ("harmonic", "morse"):
        raise ValueError("potential_type must be harmonic or morse")

    v_arr = _np.asarray(basis.V_array, _np.int64)
    J_arr = _np.asarray(basis.J_array, _np.int64)

    # M_arrayがない場合（use_M=False）はM=0で埋める
    if hasattr(basis, "M_array"):
        M_arr = _np.asarray(basis.M_array, _np.int64)
    else:
        M_arr = _np.zeros_like(J_arr, dtype=_np.int64)

    axis_idx = "xyz".index(axis_normalized)
    vib_is_morse = pot == "morse"
    vib_func = tdm_vib_morse if vib_is_morse else tdm_vib_harm
    if vib_is_morse:
        omega01_domega_to_N(basis.omega_rad_pfs, basis.delta_omega_rad_pfs)
    # ---- GPU ---------------------------------------------------------
    if backend == "cupy":
        if _cp is None:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
        mat = _dense_gpu(
            _cp.asarray(v_arr),
            _cp.asarray(J_arr),
            _cp.asarray(M_arr),
            float(mu0),
            axis_idx,
            vib_func,
        )
        return mat if dense else _csp.csr_matrix(mat)  # type: ignore[arg-type]

    # ---- CPU dense ---------------------------------------------------
    if dense:
        return _dense_core(
            v_arr,
            J_arr,
            M_arr,
            float(mu0),
            axis_idx,
            vib_is_morse,
        )

    # ---- CPU sparse --------------------------------------------------
    return _sparse_cpu(
        v_arr,
        J_arr,
        M_arr,
        float(mu0),
        axis_idx,
        vib_func,
    )
