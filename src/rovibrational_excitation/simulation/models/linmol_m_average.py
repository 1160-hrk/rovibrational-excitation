"""Fixed-linear-polarization, M-averaged linear-molecule propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.propagation.schrodinger import SchrodingerPropagator
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix

_LINEAR_POLARIZATION_TOL = 128.0 * np.finfo(np.float64).eps


def canonicalize_fixed_linear_polarization(polarization: Any) -> np.ndarray:
    """Return a real unit Jones vector, rejecting non-linear polarization."""
    vector = np.asarray(polarization, dtype=np.complex128)
    if vector.shape != (2,):
        raise ValueError("polarization must be a 2-element vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("polarization must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("polarization must be non-zero")
    vector = vector / norm

    pivot = int(np.argmax(np.abs(vector)))
    vector = vector * np.exp(-1j * np.angle(vector[pivot]))
    if np.linalg.norm(vector.imag) > _LINEAR_POLARIZATION_TOL:
        raise ValueError(
            "use_M=False requires fixed linear polarization; "
            "circular and elliptical polarization require use_M=True"
        )
    real_vector = vector.real
    return real_vector / np.linalg.norm(real_vector)


class FixedMLinMolBasis(LinMolBasis):
    """Linear-molecule basis containing one conserved magnetic quantum number."""

    def __init__(
        self,
        V_max: int,
        J_max: int,
        *,
        M: int,
        omega: float,
        B: float,
        alpha: float = 0.0,
        delta_omega: float = 0.0,
        input_units: str = "rad/fs",
        output_units: str = "J",
    ) -> None:
        if isinstance(M, bool) or not isinstance(M, (int, np.integer)):
            raise TypeError("M must be an integer")
        if abs(M) > J_max:
            raise ValueError("abs(M) must not exceed J_max")
        self.fixed_M = int(M)
        super().__init__(
            V_max,
            J_max,
            use_M=True,
            omega=omega,
            B=B,
            alpha=alpha,
            delta_omega=delta_omega,
            input_units=input_units,
            output_units=output_units,
        )

    def _generate_basis(self) -> np.ndarray:
        return np.asarray(
            [
                [v, j, self.fixed_M]
                for v in range(self.V_max + 1)
                for j in range(abs(self.fixed_M), self.J_max + 1)
            ],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class MBlockProblem:
    """One representative |M| block and its incoherent ensemble weight."""

    abs_m: int
    multiplicity: int
    weight: float
    basis: FixedMLinMolBasis
    hamiltonian: Any
    dipole: LinMolDipoleMatrix
    initial_state: np.ndarray
    reduced_indices: np.ndarray


@dataclass(frozen=True)
class MAveragePropagationResult:
    """Reduced populations plus auditable representative block trajectories."""

    time_fs: np.ndarray
    population: np.ndarray
    blocks: tuple[MBlockProblem, ...]
    block_wavefunctions: tuple[np.ndarray, ...]


def _reduced_initial_states(
    params: dict[str, Any],
) -> tuple[list[tuple[int, int]], int]:
    j_count = params["J_max"] + 1
    dimension = (params["V_max"] + 1) * j_count
    raw_indices = list(params.get("initial_states", [0]))
    if not raw_indices:
        raise ValueError("initial_states must contain at least one state index")

    indices: list[int] = []
    for raw in raw_indices:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError("initial_states must contain integer state indices")
        index = int(raw)
        if index < 0 or index >= dimension:
            raise ValueError(
                f"initial state index {index} is outside reduced dimension {dimension}"
            )
        if index not in indices:
            indices.append(index)

    states = [divmod(index, j_count) for index in indices]
    initial_j = states[0][1]
    if any(j != initial_j for _, j in states[1:]):
        raise ValueError(
            "use_M=False cannot assign an isotropic M average to a coherent "
            "superposition spanning different J values; use one J value or "
            "an explicit incoherent ensemble"
        )
    return states, initial_j


def validate_m_average_initial_states(params: dict[str, Any]) -> None:
    """Validate reduced initial-state semantics without building any matrices."""
    _reduced_initial_states(params)


def build_m_average_blocks(params: dict[str, Any]) -> tuple[MBlockProblem, ...]:
    """Build the non-negative |M| representatives for an isotropic M mixture."""
    initial_states, initial_j = _reduced_initial_states(params)
    degeneracy = 2 * initial_j + 1
    amplitude = 1.0 / np.sqrt(len(initial_states))
    sparse = params.get("sparse", not params.get("dense", True))
    dense = params.get("dense", not sparse)
    blocks: list[MBlockProblem] = []

    for abs_m in range(initial_j + 1):
        basis = FixedMLinMolBasis(
            params["V_max"],
            params["J_max"],
            M=abs_m,
            omega=params["omega_rad_phz"],
            delta_omega=params.get("delta_omega_rad_phz", 0.0),
            B=params.get("B_rad_phz", 0.0),
            alpha=params.get("alpha_rad_phz", 0.0),
            output_units="J",
            input_units="rad/fs",
        )
        initial = np.zeros(basis.size(), dtype=np.complex128)
        for v, j in initial_states:
            initial[basis.get_index((v, j, abs_m))] = amplitude

        multiplicity = 1 if abs_m == 0 else 2
        dipole = LinMolDipoleMatrix(
            basis,
            mu0=params["mu0_Cm"],
            potential_type=params.get("potential_type", "harmonic"),
            backend=params.get("backend", "numpy"),
            dense=dense,
        )
        reduced_indices = (
            basis.V_array * (params["J_max"] + 1) + basis.J_array
        ).astype(np.int64)
        blocks.append(
            MBlockProblem(
                abs_m=abs_m,
                multiplicity=multiplicity,
                weight=multiplicity / degeneracy,
                basis=basis,
                hamiltonian=basis.generate_H0(),
                dipole=dipole,
                initial_state=initial,
                reduced_indices=reduced_indices,
            )
        )
    return tuple(blocks)


def _as_numpy(array: Any) -> np.ndarray:
    getter = getattr(array, "get", None)
    if callable(getter):
        array = getter()
    return np.asarray(array)


def propagate_m_average(
    params: dict[str, Any],
    electric_field: Any,
) -> MAveragePropagationResult:
    """Propagate fixed-M blocks and incoherently sum reduced populations."""
    removed_options = {
        key for key in ("auto_timestep", "target_accuracy") if key in params
    }
    if removed_options:
        names = ", ".join(sorted(removed_options))
        raise ValueError(
            f"{names} were removed; define the ElectricField grid explicitly"
        )
    blocks = build_m_average_blocks(params)
    sparse = params.get("sparse", not params.get("dense", True))
    algorithm = params.get("algorithm", "rk4")
    propagator = SchrodingerPropagator(
        backend=params.get("backend", "numpy"),
        algorithm=algorithm,
        validate_units=params.get("validate_units", True),
        renorm=params.get("renorm", False),
        sparse=sparse,
    )
    reduced_dimension = (params["V_max"] + 1) * (params["J_max"] + 1)
    time_reference: np.ndarray | None = None
    population: np.ndarray | None = None
    trajectories: list[np.ndarray] = []

    for block in blocks:
        time_fs, wavefunction = propagator.propagate(
            hamiltonian=block.hamiltonian,
            efield=electric_field,
            dipole_matrix=block.dipole,
            initial_state=block.initial_state,
            coupling_mode="scalar",
            coupling_axis="z",
            return_traj=params.get("return_traj", True),
            return_time_psi=True,
            sample_stride=params.get("sample_stride", 1),
            nondimensional=params.get("nondimensional", False),
            verbose=params.get("verbose", False),
            algorithm=algorithm,
            sparse=sparse,
            renorm=params.get("renorm", False),
        )
        time_fs = _as_numpy(time_fs)
        wavefunction = _as_numpy(wavefunction)
        if wavefunction.ndim == 1:
            wavefunction = wavefunction.reshape(1, -1)

        if time_reference is None:
            time_reference = time_fs
            population = np.zeros(
                (wavefunction.shape[0], reduced_dimension), dtype=np.float64
            )
        elif not np.array_equal(time_fs, time_reference):
            raise RuntimeError("M blocks produced different physical output time grids")
        assert population is not None
        if wavefunction.shape[0] != population.shape[0]:
            raise RuntimeError("M blocks produced different trajectory lengths")

        block_population = np.abs(wavefunction) ** 2
        for block_index, reduced_index in enumerate(block.reduced_indices):
            population[:, reduced_index] += (
                block.weight * block_population[:, block_index]
            )
        trajectories.append(wavefunction)

    assert time_reference is not None and population is not None
    return MAveragePropagationResult(
        time_fs=time_reference,
        population=population,
        blocks=blocks,
        block_wavefunctions=tuple(trajectories),
    )
