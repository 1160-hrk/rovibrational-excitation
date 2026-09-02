"""Rigid symmetric-top basis owned by the production model package."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.basis.base import BasisBase
from rovibrational_excitation.core.operators import Hamiltonian
from rovibrational_excitation.models.parameters import SymmetricTopParameters
from rovibrational_excitation.models.symmetry import RotationalSymmetryState


class SymmetricTopBasis(BasisBase):
    """Filtered ``|v,J,K,M>`` basis for one nuclear-spin isomer."""

    quantum_number_order: ClassVar[tuple[str, ...]] = ("v", "J", "K", "M")

    def __init__(self, parameters: SymmetricTopParameters):
        if not isinstance(parameters, SymmetricTopParameters):
            raise TypeError("parameters must be SymmetricTopParameters")
        self.parameters = parameters
        states: list[tuple[int, int, int, int]] = []
        for v in range(parameters.v_max + 1):
            for j in range(parameters.j_max + 1):
                for k in range(-j, j + 1):
                    symmetry_state = RotationalSymmetryState(
                        vibrational_quantum_number=v,
                        rotational_quantum_number=j,
                        body_projection_quantum_number=k,
                        vibronic_symmetry="totally_symmetric",
                    )
                    assignment = parameters.molecule_preset.assign(
                        symmetry_state,
                        nuclear_spin_isomer=parameters.nuclear_spin_isomer,
                    )
                    if not assignment.allowed:
                        continue
                    for m in range(-j, j + 1):
                        states.append((v, j, k, m))
        if not states:
            raise ValueError(
                "selected nuclear_spin_isomer produces an empty SymTop basis "
                f"for J_max={parameters.j_max}"
            )
        self.basis = np.asarray(states, dtype=np.int64)
        self.basis.setflags(write=False)
        self.V_array = self.basis[:, 0]
        self.J_array = self.basis[:, 1]
        self.K_array = self.basis[:, 2]
        self.M_array = self.basis[:, 3]
        self.index_map = {state: index for index, state in enumerate(states)}

    def size(self) -> int:
        return int(self.basis.shape[0])

    def get_index(self, state: tuple[int, int, int, int] | Any) -> int:
        try:
            values = tuple(int(value) for value in state)
        except (TypeError, ValueError):
            raise ValueError(f"State {state!r} not found in SymTop basis") from None
        if len(values) != 4:
            raise ValueError(f"State {state!r} not found in SymTop basis")
        key = (values[0], values[1], values[2], values[3])
        try:
            return self.index_map[key]
        except KeyError:
            raise ValueError(f"State {state!r} not found in SymTop basis") from None

    def get_state(self, index: int) -> NDArray[np.int64]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("basis index must be an integer")
        if not 0 <= int(index) < self.size():
            raise ValueError(f"basis index must be between 0 and {self.size() - 1}")
        result: NDArray[np.int64] = self.basis[int(index)]
        return result

    def symmetry_state(self, index: int) -> RotationalSymmetryState:
        v, j, k, _m = map(int, self.get_state(index))
        return RotationalSymmetryState(
            vibrational_quantum_number=v,
            rotational_quantum_number=j,
            body_projection_quantum_number=k,
            vibronic_symmetry="totally_symmetric",
        )

    def generate_H0(self) -> Hamiltonian:
        parameters = self.parameters
        v = self.V_array.astype(np.float64)
        j = self.J_array.astype(np.float64)
        k = self.K_array.astype(np.float64)
        x = v + 0.5

        omega01 = parameters.vibrational_frequency.angular_rad_per_fs
        shift = parameters.anharmonic_shift.angular_rad_per_fs
        vibrational = (omega01 + shift) * x - 0.5 * shift * x**2

        b_perpendicular = (
            parameters.rotational_constant_perpendicular.angular_rad_per_fs
            - parameters.vibration_rotation_coupling_perpendicular.angular_rad_per_fs
            * x
        )
        b_parallel = (
            parameters.rotational_constant_parallel.angular_rad_per_fs
            - parameters.vibration_rotation_coupling_parallel.angular_rad_per_fs * x
        )
        frequencies = (
            vibrational
            + b_perpendicular * j * (j + 1.0)
            + (b_parallel - b_perpendicular) * k**2
        )
        information = {
            "basis_type": "SymTop",
            "quantum_number_order": self.quantum_number_order,
            "canonical_id": parameters.molecule_preset.canonical_id,
            "nuclear_spin_isomer": parameters.nuclear_spin_isomer,
            "size": self.size(),
        }
        return Hamiltonian(
            np.diag(frequencies), "rad/fs", information
        ).to_energy_units()


__all__ = ["SymmetricTopBasis"]
