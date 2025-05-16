"""
linmol_dipole/builder.py  —  高速 & ポテンシャル切替え版
=========================================================
* potential_type = 'harmonic' (従来) / 'morse'
* Morse 型の場合は生成前に
      >>> from vib_tdms.morse import omega01_domega_to_N
      >>> omega01_domega_to_N(ω01, Δω)
  で De‐Morse 定数 N をセットしておく
"""

from __future__ import annotations
from typing import Literal

import numpy as _np
import scipy.sparse as _sp
from numba import njit, prange, complex128, int64

try:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
except ImportError:
    _cp = None
    _csp = None

from rot_tdms.jm import tdm_jm_x, tdm_jm_y, tdm_jm_z
from vib_tdms.harmonic import tdm_vib_harm
from vib_tdms.morse import tdm_vib_morse

_AXIS_MAP = {"x": tdm_jm_x, "y": tdm_jm_y, "z": tdm_jm_z}
_VIB_MAP  = {"harmonic": tdm_vib_harm, "morse": tdm_vib_morse}

# ------------------------------------------------------------
# --- CPU dense ------------------------------------------------
# ------------------------------------------------------------
@njit(complex128[:, :](int64[:], int64[:], int64[:],
                       float, Literal["x","y","z"],
                       Literal["harmonic","morse"]),
      parallel=True, fastmath=True, cache=True)            # type: ignore
def _build_dense_numba(v_arr, J_arr, M_arr,
                       mu0, axis, potential):
    vib_func = _VIB_MAP[potential]
    rot_elem = _AXIS_MAP[axis]
    dim = v_arr.size
    out = _np.zeros((dim, dim), dtype=_np.complex128)

    for i in prange(dim):
        v1, J1, M1 = v_arr[i], J_arr[i], M_arr[i]
        if J1 == 0:
            continue
        for j in range(dim):
            v2, J2, M2 = v_arr[j], J_arr[j], M_arr[j]
            if J2 == 0:
                continue
            if abs(v1 - v2) != 1 or abs(J1 - J2) != 1 or abs(M1 - M2) > 1:
                continue

            r = rot_elem(J1, M1, J2, M2)
            if r == 0.0:
                continue
            vfac = vib_func(v1, v2)
            if vfac == 0.0:
                continue
            out[i, j] = mu0 * r * vfac
    return out


def _build_sparse_numba(v_arr, J_arr, M_arr,
                        mu0, axis, potential):
    vib_func = _VIB_MAP[potential]
    rot_elem = _AXIS_MAP[axis]
    data: list[complex] = []
    row: list[int] = []
    col: list[int] = []
    dim = v_arr.size

    for i in range(dim):
        v1, J1, M1 = v_arr[i], J_arr[i], M_arr[i]
        if J1 == 0:
            continue
        mask = (
            (abs(v1 - v_arr) == 1) &
            (abs(J1 - J_arr) == 1) &
            (abs(M1 - M_arr) <= 1) &
            (J_arr != 0)
        )
        for j in _np.nonzero(mask)[0]:
            r = rot_elem(J1, M1, J_arr[j], M_arr[j])
            if r == 0.0:
                continue
            vfac = vib_func(v1, v_arr[j])
            if vfac == 0.0:
                continue
            data.append(mu0 * r * vfac)
            row.append(i); col.append(j)

    return _sp.csr_matrix((data, (row, col)), shape=(dim, dim),
                          dtype=_np.complex128)

# ------------------------------------------------------------
# --- GPU -----------------------------------------------------
# ------------------------------------------------------------
def _build_cupy(v_arr, J_arr, M_arr,
                mu0, axis, potential, dense):
    xp = _cp
    vib_func = _VIB_MAP[potential]
    rot_elem = _AXIS_MAP[axis]

    v1, v2 = xp.expand_dims(v_arr,1), xp.expand_dims(v_arr,0)
    J1, J2 = xp.expand_dims(J_arr,1), xp.expand_dims(J_arr,0)
    M1, M2 = xp.expand_dims(M_arr,1), xp.expand_dims(M_arr,0)

    mask = (
        (xp.abs(v1 - v2) == 1) &
        (xp.abs(J1 - J2) == 1) &
        (xp.abs(M1 - M2) <= 1) &
        (J1 != 0) & (J2 != 0)
    )

    rot_mat = xp.vectorize(rot_elem, otypes=[xp.complex128])(J1,M1,J2,M2)
    vib_mat = xp.vectorize(vib_func, otypes=[xp.float64])(v1, v2)
    mat = mu0 * rot_mat * vib_mat
    mat *= mask

    return mat if dense else _csp.csr_matrix(mat)

# ------------------------------------------------------------
# --- Public --------------------------------------------------
# ------------------------------------------------------------
def build_mu(
    basis: "LinMolBasis",
    axis: Literal["x","y","z"],
    mu0: float,
    *,
    potential_type: Literal["harmonic","morse"] = "harmonic",
    backend: Literal["numpy","cupy"] = "numpy",
    dense: bool = True,
):
    axis = axis.lower(); potential_type = potential_type.lower()
    if axis not in "xyz": raise ValueError("axis must be x,y,z")
    if potential_type not in _VIB_MAP: raise ValueError("invalid potential_type")

    v = _np.asarray(basis.v_list, dtype=_np.int64)
    J = _np.asarray(basis.J_list, dtype=_np.int64)
    M = _np.asarray(basis.M_list, dtype=_np.int64)

    v_arr = _np.repeat(v, len(J)*len(M))
    J_arr = _np.tile(_np.repeat(J, len(M)), len(v))
    M_arr = _np.tile(M, len(v)*len(J))

    if backend == "cupy":
        if _cp is None: raise RuntimeError("CuPy not installed")
        return _build_cupy(_cp.asarray(v_arr), _cp.asarray(J_arr), _cp.asarray(M_arr),
                           float(mu0), axis, potential_type, dense)

    if dense:
        return _build_dense_numba(v_arr, J_arr, M_arr, float(mu0), axis, potential_type)
    return _build_sparse_numba(v_arr, J_arr, M_arr, float(mu0), axis, potential_type)
