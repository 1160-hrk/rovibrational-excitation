#!/usr/bin/env python
"""
check_cache.py
==============
LinMolDipoleMatrix のキャッシュ挙動を確認しつつ
μ_x 行列を pcolormesh で可視化するユーティリティ。

環境変数
--------
BACKEND=cupy   : GPU で実行（CuPy インストール必須）
DENSE=false    : CSR sparse で生成し dense に変換して描画
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix

# GPU optional --------------------------------------------------------
BACKEND = os.getenv("BACKEND", "numpy").lower()
DENSE = os.getenv("DENSE", "true").lower() == "true"

try:
    import cupy as cp
except ImportError:
    cp = None
    if BACKEND == "cupy":
        raise RuntimeError("CuPy 未インストールですが BACKEND=cupy が指定されました")


# util ----------------------------------------------------------------
def xp():
    return cp if (BACKEND == "cupy") else np


def as_np(arr):
    """NumPy or CuPy → NumPy ndarray"""
    if BACKEND == "cupy":
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        return cp.asnumpy(arr)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr)


def banner(msg):
    print(f"\n=== {msg} {'=' * (60 - len(msg))}")


# --------------------------------------------------------------------
def main():
    basis = LinMolBasis(V_max=1, J_max=1)

    dip = LinMolDipoleMatrix(
        basis,
        mu0=0.4,
        potential_type="harmonic",
        backend=BACKEND,
        dense=DENSE,
    )

    banner("First mu_x access (build)")
    t0 = time.perf_counter()
    mu_y = dip.mu_y
    mu_x = dip.mu_x
    _ = dip.mu_z
    t1 = time.perf_counter()
    print(f"  elapsed: {(t1 - t0) * 1e3:.1f} ms")

    banner("Second mu_x access (cached)")
    t0 = time.perf_counter()
    _ = dip.mu_x
    _ = dip.mu_y
    t1 = time.perf_counter()
    print(f"  elapsed: {(t1 - t0) * 1e3:.3f} ms")

    # -------------------------------------------------- pcolormesh ----
    mu_x_dense = as_np(mu_x)
    mu_y_dense = as_np(mu_y)
    print("mu_x", mu_x)
    print("mu_y", mu_y)
    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolormesh(mu_x_dense.real, cmap="RdBu_r", shading="nearest")
    plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("j index")
    ax.set_ylabel("i index")
    ax.set_title(r"$\mu_x$" f" (backend={BACKEND}, dense={DENSE})")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolormesh(
        np.real(mu_x_dense + 1j * mu_y_dense), cmap="RdBu_r", shading="nearest"
    )
    plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("j index")
    ax.set_ylabel("i index")
    ax.set_title(r"$\mu_x$" f" (backend={BACKEND}, dense={DENSE})")
    fig.tight_layout()
    plt.show()

    print("\nAll tests + plot done ✔")


if __name__ == "__main__":
    main()
