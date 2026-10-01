"""Model-independent access contract for a dipole operator."""

from __future__ import annotations

from typing import Any, Protocol


class DipoleOperator(Protocol):
    """Only the accessors required by propagation and spectroscopy.

    Results remain backend-native; this protocol does not prescribe storage,
    caching, unit-conversion implementation, or a model-specific base class.
    """

    def get_mu_in_units(
        self, axis: str, target_units: str, *, dense: bool | None = None
    ) -> Any: ...

    def get_mu_x_SI(self, *, dense: bool | None = None) -> Any: ...

    def get_mu_y_SI(self, *, dense: bool | None = None) -> Any: ...

    def get_mu_z_SI(self, *, dense: bool | None = None) -> Any: ...
