"""
rovibrational_excitation.dipole.symtop/builder.py
=================================================
Fast construction of transition-dipole matrices μ_axis (x, y, z) for
symmetric-top molecules using the SymTopBasis.

This is a *first draft* that mirrors the public API and CPU dense/sparse
implementations of the linear-molecule builder.  It relies on the
rotation matrix-element helpers tdm_jmk_{x,y,z} that include the K quantum
number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import numpy as _np
import scipy.sparse as _sp
from numba import njit, prange

try:  # Optional GPU backend (placeholder – not implemented yet)
    import cupy as _cp  # noqa: N812
    import cupyx.scipy.sparse as _csp  # noqa: N812
except ImportError:  # pragma: no cover – CPU-only fallback
    _cp = None
    _csp = None

# Rotation & vibration helpers ------------------------------------------------
from rovibrational_excitation.dipole.rot.jmk import (
    tdm_jmk_x,
    tdm_jmk_y,
    tdm_jmk_z,
)
from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm
from rovibrational_excitation.dipole.vib.morse import (
    omega01_domega_to_N,
    tdm_vib_morse,
)

if TYPE_CHECKING:
    from rovibrational_excitation.core.basis.symtop import SymTopBasis

# Runtime type alias ----------------------------------------------------------
if _cp is not None:
    Array: type = Union[_np.ndarray, _cp.ndarray]  # type: ignore
else:
    Array: type = _np.ndarray  # type: ignore

# -----------------------------------------------------------------------------
# 1. Tiny wrappers jit-compiled so that Numba kernels have no untyped globals
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def _vib_harm(v1: int, v2: int) -> float:
    """√v selection rule (Δv = ±1) – harmonic oscillator."""
    if v1 == v2 + 1:
        return _np.sqrt(v1)
    elif v2 == v1 + 1:
        return _np.sqrt(v2)
    else:
        return 0.0

_vib_morse_jit = njit(cache=True, fastmath=True)(tdm_vib_morse)  # type: ignore[arg-type]

# -----------------------------------------------------------------------------
# 2. Shared CPU dense kernel (Numba, parallel)
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _dense_core(
    v_arr,
    J_arr,
    M_arr,
    K_arr,
    mu0: float,
    axis_idx: int,
    vib_is_morse: bool,
):
    dim = v_arr.size
    out = _np.zeros((dim, dim), dtype=_np.complex128)

    for i in prange(dim):
        v1, J1, M1, K1 = v_arr[i], J_arr[i], M_arr[i], K_arr[i]
        for j in range(dim):
            v2, J2, M2, K2 = v_arr[j], J_arr[j], M_arr[j], K_arr[j]
            # --- Selection rules -------------------------------------
            if abs(J1 - J2) > 1:
                continue  # ΔJ = 0,±1 (practical: only ±1 contributes in dipole)
            if K1 != K2:
                continue  # ΔK = 0 for electric dipole within a symmetric top
            if abs(M1 - M2) > 1:
                continue  # ΔM = 0,±1

            # --- Rotational factor -----------------------------------
            if axis_idx == 0:
                rot = tdm_jmk_x(J1, M1, K1, J2, M2, K2)
            elif axis_idx == 1:
                rot = tdm_jmk_y(J1, M1, K1, J2, M2, K2)
            else:
                rot = tdm_jmk_z(J1, M1, K1, J2, M2, K2)
            if rot == 0.0:
                continue

            # --- Vibrational factor ----------------------------------
            vib = _vib_morse_jit(v1, v2) if vib_is_morse else _vib_harm(v1, v2)
            if vib == 0.0:
                continue

            out[i, j] = mu0 * rot * vib
    return out

# -----------------------------------------------------------------------------
# 3. CPU sparse implementation (Python loop – CSR)
# -----------------------------------------------------------------------------

def _sparse_cpu(v_arr, J_arr, M_arr, K_arr, mu0: float, axis_idx: int, vib_func):
    rot_funcs = (tdm_jmk_x, tdm_jmk_y, tdm_jmk_z)
    rot_func = rot_funcs[axis_idx]
    data, row, col = [], [], []

    for i, (v1, J1, M1, K1) in enumerate(zip(v_arr, J_arr, M_arr, K_arr)):
        mask = (
            (abs(J1 - J_arr) <= 1)
            & (K_arr == K1)
            & (abs(M1 - M_arr) <= 1)
        )
        for j in _np.nonzero(mask)[0]:
            rot = rot_func(J1, M1, K1, J_arr[j], M_arr[j], K_arr[j])
            if rot == 0.0:
                continue
            vib = vib_func(v1, v_arr[j])
            if vib == 0.0:
                continue
            data.append(mu0 * rot * vib)
            row.append(i)
            col.append(j)

    shape = (len(v_arr), len(v_arr))
    return _sp.csr_matrix((data, (row, col)), shape=shape, dtype=_np.complex128)

# -----------------------------------------------------------------------------
# 4. Public builder API
# -----------------------------------------------------------------------------

def build_mu(
    basis: "SymTopBasis",
    axis: Literal["x", "y", "z"],
    mu0: float,
    *,
    potential_type: Literal["harmonic", "morse"] = "harmonic",
    backend: Literal["numpy", "cupy"] = "numpy",
    dense: bool = True,
):
    """Generate transition-dipole matrix μ_axis for SymTopBasis."""
    # Normalise inputs ------------------------------------------------
    axis_norm = axis.lower()
    if axis_norm not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    pot = potential_type.lower()
    if pot not in ("harmonic", "morse"):
        raise ValueError("potential_type must be harmonic or morse")

    # Arrays ----------------------------------------------------------
    v_arr = _np.asarray(basis.V_array, _np.int64)
    J_arr = _np.asarray(basis.J_array, _np.int64)
    M_arr = _np.asarray(basis.M_array, _np.int64)
    K_arr = _np.asarray(basis.K_array, _np.int64)

    axis_idx = "xyz".index(axis_norm)
    vib_is_morse = pot == "morse"
    vib_func = tdm_vib_morse if vib_is_morse else tdm_vib_harm

    if vib_is_morse:
        omega01_domega_to_N(basis.omega_rad_pfs, basis.delta_omega_rad_pfs)

    # ----------------------------------------------------------------
    # Currently CPU-only.  GPU support can be added following LinMol.
    # ----------------------------------------------------------------
    if backend != "numpy":
        raise NotImplementedError("GPU backend not yet implemented for SymTopDipoleMatrix")

    if dense:
        return _dense_core(v_arr, J_arr, M_arr, K_arr, float(mu0), axis_idx, vib_is_morse)
    else:
        return _sparse_cpu(v_arr, J_arr, M_arr, K_arr, float(mu0), axis_idx, vib_func) 