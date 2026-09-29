#!/usr/bin/env python3
"""
吸光度スペクトル計算モジュール

密度行列から吸光度スペクトルを計算するためのクラスと関数を提供。
@core/の標準オブジェクト（Basis, Hamiltonian, DipoleMatrix）と統合。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from rovibrational_excitation.core.basis import BasisBase
from rovibrational_excitation.core.dipole import DipoleOperator
from rovibrational_excitation.core.model import CouplingMode, SystemModel
from rovibrational_excitation.core.operators import Hamiltonian
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.spectroscopy import (
    broadening,
    observables,
    response,
    transform,
)
from rovibrational_excitation.spectroscopy.conditions import (
    ExperimentalConditions,
    require_exact_units,
)
from rovibrational_excitation.spectroscopy.projection import (
    CartesianAnalyzerProjection,
    CartesianProjection,
)
from rovibrational_excitation.spectroscopy.report import (
    SpectroscopyCalculationReport,
)
from rovibrational_excitation.spectroscopy.result import ComplexResponseSpectrum

# Short aliases refer to the authoritative constants layer; no local values.
H_DIRAC = CONSTANTS.HBAR
C = CONSTANTS.C


class AbsorbanceCalculator:
    """
    密度行列から吸光度スペクトルを計算するクラス

    @core/の標準オブジェクトを使用した統一インターフェース。
    x, y, z の3軸すべての双極子モーメント成分をサポート。

    Parameters
    ----------
    basis : BasisBase
        量子基底オブジェクト
    hamiltonian : Hamiltonian
        ハミルトニアンオブジェクト（J単位推奨）
    dipole_matrix : DipoleOperator
        双極子行列オブジェクト（SI単位）
    conditions : ExperimentalConditions
        Explicit experimental conditions
    phase_matching : {'unfiltered', 'pump_probe'}
        Probe interaction前の密度行列に適用する位相整合経路。pump_probeは
        ``V_i == V_j`` のブロックを選ぶ。この選別は吸収計算前だけに
        適用され、post-probeの放射/PFID密度には適用されない。
    projection : CartesianProjection or CartesianAnalyzerProjection
        型付き偏光測定。標準吸収は前者、アナライザー複素応答は後者。

    Examples
    --------
    >>> basis = LinMolBasis(V_max=2, J_max=10, use_M=True, ...)
    >>> H0 = basis.generate_H0()
    >>> dipole = LinMolDipoleMatrix(basis=basis, mu0=1.0e-30)
    >>> calculator = AbsorbanceCalculator(
    ...     basis, H0, dipole, conditions,
    ...     phase_matching="pump_probe",
    ...     projection=CartesianProjection.from_jones(
    ...         axes="xyz", interaction=np.array([1, 0, 0])
    ...     ),
    ... )
    >>> absorbance = calculator.calculate(
    ...     rho, wavenumber, method='loop', wavenumber_units='cm^-1'
    ... )
    """

    def __init__(
        self,
        basis: BasisBase,
        hamiltonian: Hamiltonian,
        dipole_matrix: DipoleOperator,
        conditions: ExperimentalConditions,
        *,
        phase_matching: Literal["unfiltered", "pump_probe"],
        projection: CartesianProjection | CartesianAnalyzerProjection,
    ):
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.dipole_matrix = dipole_matrix
        self.phase_matching = phase_matching
        if phase_matching not in {"unfiltered", "pump_probe"}:
            raise ValueError("phase_matching must be 'unfiltered' or 'pump_probe'")
        self.conditions = conditions

        if not isinstance(
            projection,
            (CartesianProjection, CartesianAnalyzerProjection),
        ):
            raise TypeError(
                "projection must be CartesianProjection or CartesianAnalyzerProjection"
            )
        self.projection = projection
        self.axes = projection.axes_string
        self._interaction_ket = np.array(
            projection.interaction, dtype=np.complex128, copy=True
        )
        if isinstance(projection, CartesianAnalyzerProjection):
            self._detection_ket = np.array(
                projection.analyzer,
                dtype=np.complex128,
                copy=True,
            )
            self._measurement_kind = "analyzer_complex_response"
        else:
            self._detection_ket = np.array(
                projection.interaction,
                dtype=np.complex128,
                copy=True,
            )
            self._measurement_kind = "standard_absorption"

        self._setup_matrices()
        self._prepared_2d = False
        self._last_calculation_report: SpectroscopyCalculationReport | None = None
        self._last_discarded_commutator_l2_fraction = 0.0
        self._last_discarded_density_l2_fraction = 0.0

    @classmethod
    def standard_absorption(
        cls,
        model: SystemModel,
        conditions: ExperimentalConditions,
        *,
        phase_matching: Literal["unfiltered", "pump_probe"],
        projection: CartesianProjection | None = None,
    ) -> AbsorbanceCalculator:
        """Build the ordinary transmission-absorbance measurement.

        Scalar models use their model-owned storage axis and accept no Jones
        projection. Cartesian models require an explicit interaction Jones ket.
        Detection is the analyzer bra of the same physical probe ket, so for
        Hermitian Cartesian dipoles ``mu_det == mu_int.conj().T``.
        """
        if not isinstance(model, SystemModel):
            raise TypeError("model must be a SystemModel")

        if model.coupling.mode is CouplingMode.SCALAR:
            if projection is not None:
                raise ValueError(
                    "Cartesian projection is not applicable to scalar coupling"
                )
            axis = model.coupling.scalar_axis
            assert axis is not None
            projection = CartesianProjection(
                axes=(axis,),
                interaction=np.ones(1, dtype=np.complex128),
            )
        else:
            if not isinstance(projection, CartesianProjection):
                raise ValueError(
                    "CartesianProjection is required for Cartesian coupling"
                )
            model_axes = "".join(model.coupling.axes)
            if projection.axes_string != model_axes:
                raise ValueError(
                    "projection axes must exactly match model coupling axes"
                )

        return cls(
            basis=model.basis,
            hamiltonian=model.hamiltonian,
            dipole_matrix=model.dipole,
            conditions=conditions,
            phase_matching=phase_matching,
            projection=projection,
        )

    @classmethod
    def analyzer_complex_response(
        cls,
        model: SystemModel,
        conditions: ExperimentalConditions,
        *,
        phase_matching: Literal["unfiltered", "pump_probe"],
        projection: CartesianAnalyzerProjection,
    ) -> AbsorbanceCalculator:
        """Build an analyzer-projected complex-response measurement."""
        if not isinstance(model, SystemModel):
            raise TypeError("model must be a SystemModel")
        if model.coupling.mode is CouplingMode.SCALAR:
            raise ValueError("Cartesian analyzer is not applicable to scalar coupling")
        if not isinstance(projection, CartesianAnalyzerProjection):
            raise TypeError("projection must be a CartesianAnalyzerProjection")
        model_axes = "".join(model.coupling.axes)
        if projection.axes_string != model_axes:
            raise ValueError("projection axes must exactly match model coupling axes")

        return cls(
            basis=model.basis,
            hamiltonian=model.hamiltonian,
            dipole_matrix=model.dipole,
            conditions=conditions,
            phase_matching=phase_matching,
            projection=projection,
        )

    def _setup_matrices(self):
        """内部行列の準備"""
        # ハミルトニアンからエネルギー配列を取得（J単位）
        self.energy_array = self.hamiltonian.get_eigenvalues(units="J")
        self.N_level = len(self.energy_array)

        # 複素ボーア周波数行列 [rad/s - i*gamma]
        gamma_coh = self.conditions.coherence_decay_rate
        energy_vstack = np.tile(self.energy_array, (self.N_level, 1))
        self.omega_vj_vpjp_mat = (
            energy_vstack - energy_vstack.T
        ) / H_DIRAC - 1j * gamma_coh

        self._setup_phase_matching_mask()

        # 遷移双極子行列を取得
        self._setup_dipole_matrices()

    def _setup_phase_matching_mask(self) -> None:
        """Build the selected pre-probe density-matrix pathway mask."""
        if self.phase_matching == "unfiltered":
            self._phase_matching_mask = np.ones(
                (self.N_level, self.N_level),
                dtype=bool,
            )
            return

        v_array = getattr(self.basis, "V_array", None)
        if v_array is None:
            raise ValueError("phase_matching='pump_probe' requires basis.V_array")
        try:
            vibrational_levels = np.asarray(v_array, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "basis.V_array must contain finite vibrational quantum numbers"
            ) from exc
        if vibrational_levels.shape != (self.N_level,):
            raise ValueError(f"basis.V_array must have shape ({self.N_level},)")
        if not np.all(np.isfinite(vibrational_levels)):
            raise ValueError(
                "basis.V_array must contain finite vibrational quantum numbers"
            )
        self._phase_matching_mask = (
            vibrational_levels[:, np.newaxis] == vibrational_levels[np.newaxis, :]
        )

    def _validate_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Return a finite complex density matrix with the basis shape."""
        rho_array = np.asarray(rho, dtype=np.complex128)
        expected_shape = (self.N_level, self.N_level)
        if rho_array.shape != expected_shape:
            raise ValueError(
                f"rho must have shape {expected_shape}, got {rho_array.shape}"
            )
        if not np.all(np.isfinite(rho_array.real)) or not np.all(
            np.isfinite(rho_array.imag)
        ):
            raise ValueError("rho must contain only finite values")
        return rho_array

    def _select_pre_probe_density(self, rho: np.ndarray) -> np.ndarray:
        """Apply the explicit pre-probe pathway selection and record its loss."""
        selected = np.where(self._phase_matching_mask, rho, 0.0)
        total_norm = float(np.linalg.norm(rho))
        discarded_norm = float(np.linalg.norm(rho[~self._phase_matching_mask]))
        self._last_discarded_density_l2_fraction = (
            discarded_norm / total_norm if total_norm > 0.0 else 0.0
        )
        return selected

    def _setup_dipole_matrices(self):
        """Build interaction and analyzer-projected dipole operators."""
        getters = {
            "x": self.dipole_matrix.get_mu_x_SI,
            "y": self.dipole_matrix.get_mu_y_SI,
            "z": self.dipole_matrix.get_mu_z_SI,
        }
        self.mu_components: dict[str, np.ndarray] = {}
        for axis in self.axes:
            matrix = getters[axis]()
            if not isinstance(matrix, np.ndarray):
                matrix = matrix.toarray()
            component = np.asarray(matrix, dtype=np.complex128)
            expected_shape = (self.N_level, self.N_level)
            if component.shape != expected_shape:
                raise ValueError(
                    f"mu_{axis} must have shape {expected_shape}, got {component.shape}"
                )
            self.mu_components[axis] = component

        self.mu_int = np.zeros(
            (self.N_level, self.N_level),
            dtype=np.complex128,
        )
        self.mu_det = np.zeros_like(self.mu_int)
        for coefficient, axis in zip(self._interaction_ket, self.axes):
            self.mu_int += coefficient * self.mu_components[axis]
        for coefficient, axis in zip(self._detection_ket.conj(), self.axes):
            self.mu_det += coefficient * self.mu_components[axis]

        # A scale-relative roundoff threshold removes only numerical zero noise;
        # it never compares physical SI dipoles against an absolute cutoff.
        support_scale = float(np.max(np.abs(self.mu_det)))
        support_threshold = np.finfo(np.float64).eps * support_scale
        self._mu_det_support = np.abs(self.mu_det) > support_threshold
        self.ind_nonzero = np.array(np.where(self._mu_det_support))

    def prepare_2d_calculation(
        self,
        wavenumber: np.ndarray,
        *,
        wavenumber_units: str,
    ) -> None:
        """
        2D計算用の事前準備（高速化のため）

        Parameters
        ----------
        wavenumber : np.ndarray
            波数配列 [cm^-1]
        wavenumber_units : str
            必須。現在は ``"cm^-1"`` のみを受理する。
        """
        require_exact_units(
            wavenumber_units,
            name="wavenumber_units",
            expected="cm^-1",
        )
        self._prepare_2d_calculation(wavenumber)

    def _prepare_2d_calculation(self, wavenumber: np.ndarray) -> None:
        """Prepare canonical cm^-1 input for the internal 2D calculation."""
        self._omega_2d, self._one_over_denominator = response.prepare_2d_denominators(
            wavenumber,
            self.ind_nonzero,
            self.omega_vj_vpjp_mat,
        )
        self._prepared_2d = True
        self._prepared_wavenumber = np.array(wavenumber, copy=True)

    @property
    def last_calculation_report(self) -> SpectroscopyCalculationReport:
        """Return the most recent completed calculation path."""
        if self._last_calculation_report is None:
            raise RuntimeError("no spectroscopy calculation has completed")
        return self._last_calculation_report

    @staticmethod
    def _uniform_grid_spacing(grid: np.ndarray, *, name: str) -> float:
        return broadening.uniform_grid_spacing(grid, name=name)

    def _estimate_2d_bytes(self, wavenumber: np.ndarray) -> int:
        n_frequency = len(wavenumber)
        n_transition = self.ind_nonzero.shape[1]
        # Peak includes the cached complex denominator and the temporary
        # elementwise product simultaneously, plus frequency/transition vectors.
        return int(
            32 * n_frequency * n_transition + 24 * n_frequency + 16 * n_transition
        )

    def _response_entry_indices(
        self,
        commutator: np.ndarray,
        relative_threshold: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        i_indices, j_indices, discarded_fraction = response.select_response_entries(
            commutator,
            self._mu_det_support,
            relative_threshold,
        )
        self._last_discarded_commutator_l2_fraction = discarded_fraction
        return i_indices, j_indices

    @staticmethod
    def _validate_response_request(
        *,
        method: Literal[
            "matrix",
            "loop",
            "2d",
            "chunked",
            "auto",
            "approximate_sparse",
        ],
        wavenumber_units: str,
        apply_doppler: bool,
        chunk_size: int | None,
        relative_threshold: float | None,
        memory_budget_bytes: int | None,
    ) -> None:
        """Validate options shared by absorbance and complex response."""
        require_exact_units(
            wavenumber_units,
            name="wavenumber_units",
            expected="cm^-1",
        )
        valid_methods = {
            "matrix",
            "loop",
            "2d",
            "chunked",
            "auto",
            "approximate_sparse",
        }
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}")

        chunked_methods = {"chunked", "auto", "approximate_sparse"}
        if method in chunked_methods:
            if (
                not isinstance(chunk_size, int)
                or isinstance(chunk_size, bool)
                or chunk_size <= 0
            ):
                raise ValueError(
                    "chunk_size is required and must be a positive integer for "
                    "chunked, auto, and approximate_sparse methods"
                )
        elif chunk_size is not None:
            raise ValueError(
                "chunk_size is applicable only to chunked, auto, and "
                "approximate_sparse methods"
            )

        if method == "approximate_sparse":
            if relative_threshold is None:
                raise ValueError(
                    "relative_threshold is required for approximate_sparse"
                )
            if (
                not np.isfinite(relative_threshold)
                or relative_threshold <= 0.0
                or relative_threshold > 1.0
            ):
                raise ValueError("0 < relative_threshold <= 1 is required")
        elif relative_threshold is not None:
            raise ValueError(
                "relative_threshold is applicable only to approximate_sparse"
            )

        if method == "auto":
            if (
                not isinstance(memory_budget_bytes, int)
                or isinstance(memory_budget_bytes, bool)
                or memory_budget_bytes <= 0
            ):
                raise ValueError(
                    "memory_budget_bytes is required and must be a positive integer "
                    "for auto"
                )
        elif memory_budget_bytes is not None:
            raise ValueError("memory_budget_bytes is applicable only to auto")

        if apply_doppler and method not in {"matrix", "loop"}:
            raise ValueError(
                "Doppler broadening currently requires matrix or loop so every "
                "transition uses its own Doppler width"
            )

    def _execute_response_request(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        *,
        method: Literal[
            "matrix",
            "loop",
            "2d",
            "chunked",
            "auto",
            "approximate_sparse",
        ],
        apply_doppler: bool,
        require_uniform_grid: bool,
        chunk_size: int | None,
        relative_threshold: float | None,
        memory_budget_bytes: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
        """Execute one validated molecular-response request."""
        rho_array = self._select_pre_probe_density(self._validate_density_matrix(rho))
        wavenumber_array = np.asarray(wavenumber, dtype=float)
        if require_uniform_grid:
            self._uniform_grid_spacing(wavenumber_array, name="wavenumber")

        estimated_2d_bytes = self._estimate_2d_bytes(wavenumber_array)
        if method == "auto":
            assert memory_budget_bytes is not None
            executed_method = (
                "2d" if estimated_2d_bytes <= memory_budget_bytes else "chunked"
            )
        elif method == "approximate_sparse":
            executed_method = "chunked"
        else:
            executed_method = method

        self._last_discarded_commutator_l2_fraction = 0.0
        if executed_method == "chunked":
            assert chunk_size is not None
            omega, molecular_response = self._calculate_chunked(
                rho_array,
                wavenumber_array,
                chunk_size=chunk_size,
                relative_threshold=relative_threshold,
            )
        elif executed_method == "2d":
            omega, molecular_response = self._calculate_2d(rho_array, wavenumber_array)
        elif executed_method == "matrix":
            omega, molecular_response = self._calculate_matrix(
                rho_array,
                wavenumber_array,
                apply_doppler,
            )
        else:
            omega, molecular_response = self._calculate_loop(
                rho_array,
                wavenumber_array,
                apply_doppler,
            )

        return (
            wavenumber_array,
            omega,
            molecular_response,
            executed_method,
            estimated_2d_bytes,
        )

    def _record_calculation_report(
        self,
        *,
        requested_method: str,
        executed_method: str,
        estimated_2d_bytes: int,
        memory_budget_bytes: int | None,
        relative_threshold: float | None,
        device_function_applied: bool,
    ) -> SpectroscopyCalculationReport:
        """Create and retain the report for one completed request."""
        report = SpectroscopyCalculationReport(
            requested_method=requested_method,
            executed_method=executed_method,
            estimated_2d_bytes=estimated_2d_bytes,
            memory_budget_bytes=memory_budget_bytes,
            relative_threshold=relative_threshold,
            discarded_commutator_l2_fraction=(
                self._last_discarded_commutator_l2_fraction
            ),
            phase_matching=self.phase_matching,
            discarded_density_l2_fraction=(self._last_discarded_density_l2_fraction),
            device_function_applied=device_function_applied,
        )
        self._last_calculation_report = report
        return report

    def calculate(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        method: Literal[
            "matrix",
            "loop",
            "2d",
            "chunked",
            "auto",
            "approximate_sparse",
        ],
        *,
        wavenumber_units: str,
        apply_doppler: bool = False,
        apply_device_function: bool = False,
        device_resolution: float | None = None,
        device_resolution_units: str | None = None,
        chunk_size: int | None = None,
        relative_threshold: float | None = None,
        memory_budget_bytes: int | None = None,
    ) -> np.ndarray:
        """Calculate absorbance from a wavenumber grid explicitly in cm^-1.

        ``device_resolution`` and ``device_resolution_units`` must be supplied
        together exactly when ``apply_device_function=True``.
        """
        self._require_standard_absorption()
        self._validate_response_request(
            method=method,
            wavenumber_units=wavenumber_units,
            apply_doppler=apply_doppler,
            chunk_size=chunk_size,
            relative_threshold=relative_threshold,
            memory_budget_bytes=memory_budget_bytes,
        )
        if apply_device_function:
            if (
                device_resolution is None
                or device_resolution_units is None
                or not np.isfinite(device_resolution)
                or device_resolution <= 0.0
            ):
                raise ValueError(
                    "finite positive device_resolution and device_resolution_units "
                    "are required when apply_device_function=True"
                )
            require_exact_units(
                device_resolution_units,
                name="device_resolution_units",
                expected="cm^-1",
            )
        elif device_resolution is not None or device_resolution_units is not None:
            raise ValueError(
                "device_resolution and device_resolution_units are applicable only "
                "when apply_device_function=True"
            )

        (
            wavenumber_array,
            omega,
            molecular_response,
            executed_method,
            estimated_2d_bytes,
        ) = self._execute_response_request(
            rho,
            wavenumber,
            method=method,
            apply_doppler=apply_doppler,
            require_uniform_grid=apply_doppler or apply_device_function,
            chunk_size=chunk_size,
            relative_threshold=relative_threshold,
            memory_budget_bytes=memory_budget_bytes,
        )
        spectrum = self._response_to_absorbance(omega, molecular_response)
        if apply_device_function:
            assert device_resolution is not None
            assert device_resolution_units is not None
            spectrum = self.apply_device_function(
                spectrum,
                wavenumber_array,
                resolution=device_resolution,
                wavenumber_units=wavenumber_units,
                resolution_units=device_resolution_units,
            )

        self._record_calculation_report(
            requested_method=method,
            executed_method=executed_method,
            estimated_2d_bytes=estimated_2d_bytes,
            memory_budget_bytes=memory_budget_bytes,
            relative_threshold=relative_threshold,
            device_function_applied=apply_device_function,
        )
        return spectrum

    def calculate_complex_response(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        method: Literal[
            "matrix",
            "loop",
            "2d",
            "chunked",
            "auto",
            "approximate_sparse",
        ],
        *,
        wavenumber_units: str,
        apply_doppler: bool = False,
        chunk_size: int | None = None,
        relative_threshold: float | None = None,
        memory_budget_bytes: int | None = None,
    ) -> ComplexResponseSpectrum:
        """Return the projected complex per-molecule response before mOD."""
        self._validate_response_request(
            method=method,
            wavenumber_units=wavenumber_units,
            apply_doppler=apply_doppler,
            chunk_size=chunk_size,
            relative_threshold=relative_threshold,
            memory_budget_bytes=memory_budget_bytes,
        )
        (
            wavenumber_array,
            _omega,
            molecular_response,
            executed_method,
            estimated_2d_bytes,
        ) = self._execute_response_request(
            rho,
            wavenumber,
            method=method,
            apply_doppler=apply_doppler,
            require_uniform_grid=apply_doppler,
            chunk_size=chunk_size,
            relative_threshold=relative_threshold,
            memory_budget_bytes=memory_budget_bytes,
        )
        report = self._record_calculation_report(
            requested_method=method,
            executed_method=executed_method,
            estimated_2d_bytes=estimated_2d_bytes,
            memory_budget_bytes=memory_budget_bytes,
            relative_threshold=relative_threshold,
            device_function_applied=False,
        )
        return ComplexResponseSpectrum(
            wavenumber_cm_inverse=wavenumber_array,
            molecular_response_c2_m2_per_j=molecular_response,
            calculation_report=report,
        )

    def _calculate_2d(
        self, rho: np.ndarray, wavenumber: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the cached two-dimensional response route."""
        if not self._prepared_2d or not np.array_equal(
            wavenumber, self._prepared_wavenumber
        ):
            self._prepare_2d_calculation(wavenumber)

        response_sum = response.calculate_2d_response(
            rho,
            self.mu_int,
            self.mu_det,
            self.ind_nonzero,
            self._one_over_denominator,
        )
        omega = self._omega_2d[:, 0]
        return omega, response_sum

    def _calculate_matrix(
        self, rho: np.ndarray, wavenumber: np.ndarray, apply_doppler: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the exact list-and-sum response route."""
        omega, response_sum = response.calculate_matrix_response(
            rho,
            wavenumber,
            self.mu_int,
            self.mu_det,
            self.ind_nonzero,
            self.omega_vj_vpjp_mat,
            apply_doppler=apply_doppler,
            doppler_broadener=self._apply_doppler_broadening,
        )
        return omega, response_sum

    def _calculate_loop(
        self, rho: np.ndarray, wavenumber: np.ndarray, apply_doppler: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the exact in-place accumulation response route."""
        omega, response_sum = response.calculate_loop_response(
            rho,
            wavenumber,
            self.mu_int,
            self.mu_det,
            self.ind_nonzero,
            self.omega_vj_vpjp_mat,
            apply_doppler=apply_doppler,
            doppler_broadener=self._apply_doppler_broadening,
        )
        return omega, response_sum

    def _calculate_chunked(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        chunk_size: int,
        relative_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the exact or explicitly approximate chunked route."""
        commutator = response.sparse_commutator(self.mu_int, rho)
        i_indices, j_indices = self._response_entry_indices(
            commutator,
            relative_threshold,
        )

        omega, response_sum = response.calculate_chunked_response(
            commutator,
            wavenumber,
            i_indices,
            j_indices,
            self.mu_det,
            self.omega_vj_vpjp_mat,
            chunk_size=chunk_size,
        )
        return omega, response_sum

    def _response_to_absorbance(
        self, omega: np.ndarray, response: np.ndarray
    ) -> np.ndarray:
        """Convert the molecular response to absorbance in mOD."""
        return observables.response_to_absorbance(
            omega,
            response,
            number_density=self.conditions.number_density,
            optical_length_m=self.conditions.optical_length_m,
        )

    @staticmethod
    def _filter_complex_gaussian(
        response: np.ndarray,
        sigma_pixels: float,
    ) -> np.ndarray:
        return broadening.filter_complex_gaussian(response, sigma_pixels)

    def _apply_doppler_broadening(
        self,
        omega: np.ndarray,
        response: np.ndarray,
        omega0: float,
    ) -> np.ndarray:
        return broadening.apply_doppler_broadening(
            omega,
            response,
            omega0,
            temperature_k=self.conditions.temperature_k,
            molecular_mass_kg=self.conditions.molecular_mass_kg,
        )

    def _require_standard_absorption(self) -> None:
        if self._measurement_kind == "analyzer_complex_response":
            raise ValueError(
                "analyzer complex response requires a complex-response observable; "
                "scalar mOD conversion is undefined"
            )

    def calculate_radiation_spectrum(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        *,
        wavenumber_units: str,
    ) -> np.ndarray:
        """
        放射スペクトルを計算（例：PFID）

        密度行列の非対角要素から直接放射を計算

        Parameters
        ----------
        rho : np.ndarray
            密度行列（コヒーレンスを含む）
        wavenumber : np.ndarray
            波数配列 [cm^-1]
        wavenumber_units : str
            必須。現在は ``"cm^-1"`` のみを受理する。

        Returns
        -------
        np.ndarray
            放射スペクトル [mOD]
        """
        self._require_standard_absorption()
        require_exact_units(
            wavenumber_units,
            name="wavenumber_units",
            expected="cm^-1",
        )
        rho_array = self._validate_density_matrix(rho)
        wavenumber_array = np.asarray(wavenumber, dtype=float)
        omega = 2 * np.pi * C * 1e2 * wavenumber_array

        resp_lin_per_mole = transform.radiation_response(
            rho_array,
            omega,
            self.ind_nonzero,
            self.mu_det,
            self.omega_vj_vpjp_mat,
        )

        return self._response_to_absorbance(omega, resp_lin_per_mole)

    def calculate_pfid_spectrum(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        *,
        wavenumber_units: str,
    ) -> np.ndarray:
        """
        Probe-induced free induction decay (PFID) スペクトルを計算

        プローブパルス後の自由誘導減衰からのスペクトル

        Parameters
        ----------
        rho : np.ndarray
            プローブ相互作用後の密度行列
        wavenumber : np.ndarray
            波数配列 [cm^-1]
        wavenumber_units : str
            必須。現在は ``"cm^-1"`` のみを受理する。

        Returns
        -------
        np.ndarray
            PFIDスペクトル [mOD]
        """
        # PFIDは放射スペクトルと同じ計算
        return self.calculate_radiation_spectrum(
            rho,
            wavenumber,
            wavenumber_units=wavenumber_units,
        )

    def apply_device_function(
        self,
        spectrum: np.ndarray,
        wavenumber: np.ndarray,
        resolution: float,
        *,
        wavenumber_units: str,
        resolution_units: str,
        function_type: Literal["sinc", "sinc2", "gaussian"] = "sinc2",
    ) -> np.ndarray:
        """Apply a normalized instrument response on an explicit cm^-1 grid."""
        return broadening.apply_device_function(
            spectrum,
            wavenumber,
            resolution,
            wavenumber_units=wavenumber_units,
            resolution_units=resolution_units,
            function_type=function_type,
        )


# ヘルパー関数
def create_calculator_from_params(
    basis: BasisBase,
    hamiltonian: Hamiltonian,
    dipole_matrix: DipoleOperator,
    *,
    temperature: float,
    temperature_units: str,
    pressure: float,
    pressure_units: str,
    optical_length: float,
    optical_length_units: str,
    coherence_time: float,
    coherence_time_units: str,
    molecular_mass: float,
    molecular_mass_units: str,
    phase_matching: Literal["unfiltered", "pump_probe"],
    projection: CartesianProjection | CartesianAnalyzerProjection,
) -> AbsorbanceCalculator:
    """
    パラメータから計算機を作成するヘルパー関数

    Parameters
    ----------
    basis : BasisBase
        量子基底
    hamiltonian : Hamiltonian
        ハミルトニアン
    dipole_matrix : DipoleOperator
        双極子行列
    temperature, temperature_units
        温度と必須単位。現在は ``K`` のみ。
    pressure, pressure_units
        圧力と必須単位。現在は ``Pa`` のみ。
    optical_length, optical_length_units
        光路長と必須単位。現在は ``m`` のみ。
    coherence_time, coherence_time_units
        コヒーレンス緩和時間と必須単位。現在は ``ps`` のみ。
    molecular_mass, molecular_mass_units
        1分子あたりの質量と必須単位。現在は ``kg`` のみ。
    phase_matching : {'unfiltered', 'pump_probe'}
        Probe interaction前の密度行列に適用する位相整合経路
    projection : CartesianProjection or CartesianAnalyzerProjection
        必須の型付き偏光測定。

    Returns
    -------
    AbsorbanceCalculator
        初期化された計算機オブジェクト
    """
    conditions = ExperimentalConditions(
        temperature=temperature,
        temperature_units=temperature_units,
        pressure=pressure,
        pressure_units=pressure_units,
        optical_length=optical_length,
        optical_length_units=optical_length_units,
        coherence_time=coherence_time,
        coherence_time_units=coherence_time_units,
        molecular_mass=molecular_mass,
        molecular_mass_units=molecular_mass_units,
    )

    return AbsorbanceCalculator(
        basis=basis,
        hamiltonian=hamiltonian,
        dipole_matrix=dipole_matrix,
        conditions=conditions,
        phase_matching=phase_matching,
        projection=projection,
    )
