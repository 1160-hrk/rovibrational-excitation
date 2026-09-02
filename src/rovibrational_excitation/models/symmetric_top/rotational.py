"""Rank-one direction-cosine elements for a parallel symmetric-top band."""

from __future__ import annotations

import math
from functools import cache


def _phase(integer: int) -> float:
    return -1.0 if integer % 2 else 1.0


@cache
def _wigner_3j_integer(
    j1: int,
    j2: int,
    j3: int,
    m1: int,
    m2: int,
    m3: int,
) -> float:
    """Evaluate an integer Wigner-3j symbol with the Racah formula."""
    if (
        min(j1, j2, j3) < 0
        or abs(m1) > j1
        or abs(m2) > j2
        or abs(m3) > j3
        or m1 + m2 + m3 != 0
        or j3 > j1 + j2
        or j3 < abs(j1 - j2)
    ):
        return 0.0

    log_triangle = (
        math.lgamma(j1 + j2 - j3 + 1)
        + math.lgamma(j1 - j2 + j3 + 1)
        + math.lgamma(-j1 + j2 + j3 + 1)
        - math.lgamma(j1 + j2 + j3 + 2)
    )
    log_m_factorials = sum(
        math.lgamma(value + 1)
        for value in (
            j1 + m1,
            j1 - m1,
            j2 + m2,
            j2 - m2,
            j3 + m3,
            j3 - m3,
        )
    )
    prefactor = _phase(j1 - j2 - m3) * math.exp(0.5 * (log_triangle + log_m_factorials))

    z_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    z_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    if z_min > z_max:
        return 0.0

    series = 0.0
    for z in range(z_min, z_max + 1):
        denominator_arguments = (
            z,
            j1 + j2 - j3 - z,
            j1 - m1 - z,
            j2 + m2 - z,
            j3 - j2 + m1 + z,
            j3 - j1 - m2 + z,
        )
        log_denominator = sum(
            math.lgamma(argument + 1) for argument in denominator_arguments
        )
        series += _phase(z) * math.exp(-log_denominator)
    return prefactor * series


def parallel_spherical_direction_cosine(
    j_bra: int,
    k_bra: int,
    m_bra: int,
    j_ket: int,
    k_ket: int,
    m_ket: int,
    p: int,
) -> float:
    """Return ``<J'K'M'|D^1_{p0}*|JKM>`` for a parallel band."""
    if p not in {-1, 0, 1}:
        raise ValueError("p must be -1, 0, or 1")
    if k_bra != k_ket:
        return 0.0
    return (
        _phase(m_bra - k_ket)
        * math.sqrt((2 * j_bra + 1) * (2 * j_ket + 1))
        * _wigner_3j_integer(j_bra, 1, j_ket, -m_bra, p, m_ket)
        * _wigner_3j_integer(j_bra, 1, j_ket, -k_bra, 0, k_ket)
    )


def parallel_cartesian_direction_cosine(
    axis: str,
    j_bra: int,
    k_bra: int,
    m_bra: int,
    j_ket: int,
    k_ket: int,
    m_ket: int,
) -> complex:
    """Return one lab-frame Cartesian direction-cosine matrix element."""
    if axis == "z":
        return complex(
            parallel_spherical_direction_cosine(
                j_bra,
                k_bra,
                m_bra,
                j_ket,
                k_ket,
                m_ket,
                0,
            )
        )
    plus = parallel_spherical_direction_cosine(
        j_bra,
        k_bra,
        m_bra,
        j_ket,
        k_ket,
        m_ket,
        1,
    )
    minus = parallel_spherical_direction_cosine(
        j_bra,
        k_bra,
        m_bra,
        j_ket,
        k_ket,
        m_ket,
        -1,
    )
    if axis == "x":
        return complex(-(plus - minus) / math.sqrt(2.0))
    if axis == "y":
        return complex(1j * (plus + minus) / math.sqrt(2.0))
    raise ValueError("axis must be x, y, or z")


__all__ = [
    "parallel_cartesian_direction_cosine",
    "parallel_spherical_direction_cosine",
]
