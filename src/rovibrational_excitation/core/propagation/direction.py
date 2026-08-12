"""Explicit integration direction for reversible state propagation."""

from __future__ import annotations

from enum import Enum


class PropagationDirection(Enum):
    """Direction in which the numerical integration traverses physical time."""

    FORWARD = 1.0
    BACKWARD = -1.0

    @property
    def sign(self) -> float:
        """Signed multiplier applied to the positive propagation interval."""
        return float(self.value)


__all__ = ["PropagationDirection"]
