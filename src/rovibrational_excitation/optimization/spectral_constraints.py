"""Historical frequency-domain filter for ``legacy_batch_overlap``.

The helper constructs a nonnegative dimensionless penalty ``alpha`` on an
rFFT grid and solves ``U[k] = S[k] / (1 + alpha[k])``.  This algebra is
independently referenced, but it does not by itself establish monotonic
convergence for standard Krotov control.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Literal, cast

import numpy as np

from rovibrational_excitation.core.units.converters import converter

SpectralMode = Literal["pass", "stop"]
SpectralCombine = Literal["max", "sum"]

_SPECTRUM_REQUIRED = {
    "method",
    "bands",
    "units",
    "mode",
    "combine",
    "fwhm",
    "alpha_scale",
}
_SPECTRUM_OPTIONAL = {"weights"}


@dataclass(frozen=True, slots=True)
class LegacySpectralFilter:
    """One historical penalty mask compiled on an exact rFFT grid."""

    alpha_mask: np.ndarray

    def apply(self, source: np.ndarray) -> np.ndarray:
        return solve_update_in_frequency(source, self.alpha_mask)


@dataclass(frozen=True, slots=True)
class LegacySpectralConstraint:
    """Validated configuration for the legacy batch-overlap filter only."""

    bands: tuple[tuple[float, float], ...]
    units: str
    mode: SpectralMode
    combine: SpectralCombine
    fwhm: bool
    weights: tuple[float, ...] | None
    alpha_scale: float

    def compile(self, freq_phz: np.ndarray) -> LegacySpectralFilter:
        return LegacySpectralFilter(
            build_alpha_mask(
                freq_phz,
                self.bands,
                units=self.units,
                mode=self.mode,
                combine=self.combine,
                fwhm=self.fwhm,
                weights=self.weights,
                alpha_scale=self.alpha_scale,
            )
        )


def _names(values: set[Any]) -> str:
    return ", ".join(
        sorted(value if isinstance(value, str) else repr(value) for value in values)
    )


def _sequence(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence") from exc


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def parse_legacy_spectral_constraint(value: Any) -> LegacySpectralConstraint:
    """Validate and freeze the legacy-only spectral-constraint mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("spectrum_constraints must be a mapping")
    keys = set(value)
    unknown = keys - (_SPECTRUM_REQUIRED | _SPECTRUM_OPTIONAL)
    if unknown:
        raise ValueError("unsupported spectrum_constraints options: " + _names(unknown))
    missing = _SPECTRUM_REQUIRED - keys
    if missing:
        raise ValueError(
            "missing required spectrum_constraints options: " + _names(missing)
        )
    if value["method"] != "monotonic_kernel":
        raise ValueError("spectrum_constraints.method must be 'monotonic_kernel'")

    raw_bands = _sequence(value["bands"], label="spectrum_constraints.bands")
    if not raw_bands:
        raise ValueError("spectrum_constraints.bands must be nonempty")
    units = value["units"]
    if not isinstance(units, str):
        raise ValueError(
            "spectrum_constraints.units must be a supported frequency unit"
        )
    try:
        converter.convert_frequency(1.0, units, "PHz")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "spectrum_constraints.units must be a supported frequency unit"
        ) from exc

    bands: list[tuple[float, float]] = []
    for index, raw_band in enumerate(raw_bands):
        band = _sequence(raw_band, label=f"spectrum_constraints.bands[{index}]")
        if len(band) != 2:
            raise ValueError(
                f"spectrum_constraints.bands[{index}] must be [center, width]"
            )
        center = _finite_real(
            band[0], label=f"spectrum_constraints.bands[{index}] center"
        )
        width = _finite_real(
            band[1], label=f"spectrum_constraints.bands[{index}] width"
        )
        try:
            converter.convert_frequency(center, units, "PHz")
            converted_width = float(converter.convert_frequency(width, units, "PHz"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid spectrum_constraints.bands[{index}] frequency"
            ) from exc
        if converted_width <= 0.0:
            raise ValueError(
                f"spectrum_constraints.bands[{index}] width must be positive"
            )
        bands.append((center, width))

    mode = value["mode"]
    if not isinstance(mode, str) or mode not in {"pass", "stop"}:
        raise ValueError("spectrum_constraints.mode must be one of: pass, stop")
    combine = value["combine"]
    if not isinstance(combine, str) or combine not in {"max", "sum"}:
        raise ValueError("spectrum_constraints.combine must be one of: max, sum")
    fwhm = value["fwhm"]
    if not isinstance(fwhm, bool):
        raise ValueError("spectrum_constraints.fwhm must be a bool")
    alpha_scale = _finite_real(
        value["alpha_scale"], label="spectrum_constraints.alpha_scale"
    )
    if alpha_scale < 0.0:
        raise ValueError("spectrum_constraints.alpha_scale must be nonnegative")

    weights: tuple[float, ...] | None = None
    if "weights" in value:
        if combine != "sum":
            raise ValueError(
                "spectrum_constraints.weights is accepted only with combine='sum'"
            )
        if value["weights"] is not None:
            raw_weights = _sequence(
                value["weights"], label="spectrum_constraints.weights"
            )
            if len(raw_weights) != len(raw_bands):
                raise ValueError("spectrum_constraints.weights length must match bands")
            converted_weights: list[float] = []
            for raw_weight in raw_weights:
                weight = _finite_real(
                    raw_weight, label="spectrum_constraints.weights entries"
                )
                if weight < 0.0:
                    raise ValueError(
                        "spectrum_constraints.weights entries must be nonnegative"
                    )
                converted_weights.append(weight)
            weights = tuple(converted_weights)

    return LegacySpectralConstraint(
        bands=tuple(bands),
        units=units,
        mode=cast(SpectralMode, mode),
        combine=cast(SpectralCombine, combine),
        fwhm=fwhm,
        weights=weights,
        alpha_scale=alpha_scale,
    )


def _to_phz(x: float | np.ndarray, units: str) -> float | np.ndarray:
    """任意の周波数単位から PHz（cycles/fs）へ変換。

    units: "cm^-1" | "PHz" | "rad/fs" などを想定。
    """
    if units == "PHz":
        return x
    return converter.convert_frequency(x, units, "PHz")


def _fwhm_to_sigma(fwhm: float) -> float:
    """ガウシアンの FWHM を標準偏差 σ に変換。"""
    return float(float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0))))


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
                raise ValueError(
                    "weights length must match bands length for combine='sum'"
                )
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
    # rFFT length and penalty-domain validation.
    n_rfft = N // 2 + 1
    raw_alpha = np.asarray(alpha_mask)
    if np.iscomplexobj(raw_alpha) or np.issubdtype(raw_alpha.dtype, np.bool_):
        raise ValueError("alpha_mask must be a finite nonnegative real vector")
    try:
        alpha = np.asarray(raw_alpha, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha_mask must be a finite nonnegative real vector") from exc
    if alpha.ndim != 1 or alpha.shape[0] != n_rfft:
        raise ValueError("alpha_mask length must be N//2+1 for rFFT")
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.0):
        raise ValueError("alpha_mask must be finite and nonnegative")

    out = np.zeros_like(s, dtype=float)
    denom = 1.0 + alpha
    for k in range(ncomp):
        S_hat = np.fft.rfft(s[:, k])
        U_hat = S_hat / denom
        u = np.fft.irfft(U_hat, n=N)
        out[:, k] = np.real(u)
    return out.reshape(source.shape)
