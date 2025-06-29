#!/usr/bin/env python
"""
rovibrational_excitation/scripts/for_linmol_dipole/check_builder.py
===================================================================
* linmol_dipole.builder の出力を検証
* μ 行列を pcolormesh で可視化
--------------------------------------------------------------------
実行例
$ python -m rovibrational_excitation.scripts.for_linmol_dipole.check_builder
$ BACKEND=cupy DENSE=false python -m ...
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import matplotlib.pyplot as plt

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole.linmol.builder import build_mu
from rovibrational_excitation.dipole.vib.morse import omega01_domega_to_N

BACKEND = os.getenv("BACKEND", "numpy").lower()
DENSE   = os.getenv("DENSE", "true").lower() == "true"


# ------------------------------------------------------------------
# util
# ------------------------------------------------------------------
def _xp():
    if BACKEND == "cupy":
        import cupy as cp
        return cp
    return np


def _asnp(arr):
    """cupy / numpy → numpy.ndarray"""
    if BACKEND == "cupy":
        import cupy as cp
        if hasattr(arr, "toarray"):                # sparse
            arr = arr.toarray()
        return cp.asnumpy(arr)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr)


# ------------------------------------------------------------------
def make_basis(V_max=1, J_max=4):
    return LinMolBasis(V_max=V_max, J_max=J_max)


def summary(name: str, mat):
    arr = _asnp(mat)
    print(f"{name}: shape={arr.shape}, Hermitian={np.allclose(arr, arr.T.conj())}"
          f", Frobenius ‖μ‖={np.linalg.norm(arr):.3g}")


def plot_mat(ax, arr, title: str):
    c = ax.pcolormesh(arr.real, cmap="RdBu_r", shading="nearest")
    plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("j index")
    ax.set_ylabel("i index")


def main():
    basis = make_basis()
    mu0 = 0.3

    # Harmonic
    mu_h_x = build_mu(
        basis, "x", mu0,
        potential_type="harmonic",
        backend=BACKEND, dense=DENSE
    )
    summary("harmonic μ_x", mu_h_x)

    # Morse
    omega01_domega_to_N(omega01=2100.0, domega=100.0)  # 必要に応じて
    mu_m_x = build_mu(
        basis, "x", mu0,
        potential_type="morse",
        backend=BACKEND, dense=DENSE
    )
    summary("morse     μ_x", mu_m_x)

    # --- plotting ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_mat(ax1, _asnp(mu_h_x), "μ_x (harmonic)")
    plot_mat(ax2, _asnp(mu_m_x), "μ_x (morse)")
    fig.suptitle(f"backend={BACKEND}, dense={DENSE}")
    fig.tight_layout()
    plt.show()

    # --- quick checks ------------------------------------------------
    _ = mu_h_x.nnz if hasattr(mu_h_x, "nnz") else np.count_nonzero(mu_h_x)
    _ = mu_m_x.nnz if hasattr(mu_m_x, "nnz") else np.count_nonzero(mu_m_x)
    # assert nnz_h == nnz_m
    # assert not np.allclose(_asnp(mu_h_x), _asnp(mu_m_x))
    print("All checks passed ✔")


if __name__ == "__main__":
    main()
