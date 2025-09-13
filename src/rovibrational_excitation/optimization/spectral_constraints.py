"""
スペクトル制約（Krotov 単調収束対応）
====================================

論文に準拠したスペクトル制約を、周波数領域の正則化として実装する補助関数群。

- α(ω) ≥ 0 を周波数グリッド上で構築（ガウシアン帯域の合成）
- 源項 s(t) の FFT を取り、U(ω) = S(ω) / (1+α(ω)) で更新量を解く

注意:
- 周波数グリッドは rFFT 用の半分スペクトル (N//2+1) を想定
- 周波数単位は PHz（= cycles/fs）に統一する
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np

from rovibrational_excitation.core.units.converters import converter


def _to_phz(x: float | np.ndarray, units: str) -> float | np.ndarray:
    """任意の周波数単位から PHz（cycles/fs）へ変換。

    units: "cm^-1" | "PHz" | "rad/fs" などを想定。
    """
    if units == "PHz":
        return x
    return converter.convert_frequency(x, units, "PHz")


def _fwhm_to_sigma(fwhm: float) -> float:
    """ガウシアンの FWHM を標準偏差 σ に変換。"""
    return float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def build_alpha_mask(
    freq_phz: np.ndarray,
    bands: Sequence[Sequence[float]] | np.ndarray,
    *,
    units: str = "cm^-1",
    mode: str = "pass",
    combine: str = "max",
    fwhm: bool = True,
    weights: Iterable[float] | None = None,
    alpha_scale: float = 1.0,
) -> np.ndarray:
    """
    周波数グリッド上に α(ω) を構築。

    Parameters
    ----------
    freq_phz : np.ndarray
        rFFTFreq で得た周波数（PHz = cycles/fs）
    bands : list[[center, width], ...]
        ガウシアン中心と幅。幅は FWHM か σ（fwhm フラグで解釈）
    units : str
        bands の単位（"cm^-1" | "PHz" | "rad/fs" など）
    mode : str
        "pass" → 通過帯域を 0 ペナルティ（α=1-G）、"stop" → 帯域をペナルティ（α=G）
    combine : str
        "max" で帯域の最大、"sum" で重み付け和（[0,1] にクリップ）
    fwhm : bool
        True なら width を FWHM として σ に変換
    weights : Iterable[float] | None
        combine="sum" のときに使用する重み（長さは bands と一致すること）
    alpha_scale : float
        最終 α(ω) のスケール係数（既定 1.0）

    Returns
    -------
    np.ndarray
        α(ω) の配列（長さ len(freq_phz)）
    """
    freq = np.asarray(freq_phz, dtype=float)
    nb = len(bands)
    if nb == 0:
        return np.zeros_like(freq)

    centers_phz = np.empty(nb, dtype=float)
    sigmas_phz = np.empty(nb, dtype=float)

    for i, bw in enumerate(bands):
        if len(bw) != 2:
            raise ValueError("bands entries must be [center, width]")
        c, w = float(bw[0]), float(bw[1])
        c_phz = float(_to_phz(c, units))
        w_phz = float(_to_phz(w, units))
        sigma = _fwhm_to_sigma(w_phz) if fwhm else w_phz
        if sigma <= 0.0:
            raise ValueError("band width must be positive")
        centers_phz[i] = c_phz
        sigmas_phz[i] = sigma

    # 各帯域のガウシアンを合成
    if combine not in ("max", "sum"):
        raise ValueError("combine must be 'max' or 'sum'")
    acc = np.zeros_like(freq)
    if combine == "max":
        acc[:] = 0.0
        for c, s in zip(centers_phz, sigmas_phz):
            acc = np.maximum(acc, np.exp(-0.5 * ((freq - c) / s) ** 2))
    else:  # sum
        if weights is None:
            wts = np.ones(nb, dtype=float)
        else:
            wts_arr = np.asarray(list(weights), dtype=float)
            if wts_arr.size != nb:
                raise ValueError("weights length must match bands length for combine='sum'")
            wts = wts_arr
        for c, s, w in zip(centers_phz, sigmas_phz, wts):
            acc += float(w) * np.exp(-0.5 * ((freq - c) / s) ** 2)
        acc = np.clip(acc, 0.0, 1.0)

    if mode not in ("pass", "stop"):
        raise ValueError("mode must be 'pass' or 'stop'")

    if mode == "pass":
        # 通過帯域 → ペナルティを 0 へ
        alpha_raw = 1.0 - acc
    else:  # stop
        alpha_raw = acc

    alpha = alpha_scale * np.maximum(alpha_raw, 0.0)
    return alpha.astype(float, copy=False)


def solve_update_in_frequency(source: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """
    源項 s(t) から更新量 u(t) を周波数領域で解く。

    各成分について:
      Û(ω) = Ŝ(ω) / (1 + α(ω))
      u(t) = irfft(Û)

    Parameters
    ----------
    source : np.ndarray
        時間領域の源項（形状: (N, 2) など）。最後の次元が偏光成分。
    alpha_mask : np.ndarray
        rFFT の周波数長 (N//2 + 1) の α(ω)

    Returns
    -------
    np.ndarray
        時間領域の更新量（source と同形状）
    """
    s = np.asarray(source)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    N = s.shape[0]
    ncomp = s.shape[1]
    # rFFT 長の検証
    n_rfft = N // 2 + 1
    if alpha_mask.shape[0] != n_rfft:
        raise ValueError("alpha_mask length must be N//2+1 for rFFT")

    out = np.zeros_like(s, dtype=float)
    denom = 1.0 + np.asarray(alpha_mask, dtype=float)
    denom = np.maximum(denom, 1e-16)
    for k in range(ncomp):
        S_hat = np.fft.rfft(s[:, k])
        U_hat = S_hat / denom
        u = np.fft.irfft(U_hat, n=N)
        out[:, k] = np.real(u)
    return out.reshape(source.shape)

