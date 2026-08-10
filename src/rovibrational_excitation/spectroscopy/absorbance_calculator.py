#!/usr/bin/env python3
"""
吸光度スペクトル計算モジュール

密度行列から吸光度スペクトルを計算するためのクラスと関数を提供。
@core/の標準オブジェクト（Basis, Hamiltonian, DipoleMatrix）と統合。
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from rovibrational_excitation.core.basis import BasisBase
from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.dipole.base import DipoleMatrixBase

# Short aliases refer to the authoritative constants layer; no local values.
H_DIRAC = CONSTANTS.HBAR
C = CONSTANTS.C
EPS = CONSTANTS.EPSILON0
KB = CONSTANTS.BOLTZMANN


@dataclass(frozen=True, slots=True)
class SpectroscopyCalculationReport:
    """Observable record of the numerical spectroscopy path used."""

    requested_method: str
    executed_method: str
    estimated_2d_bytes: int
    memory_budget_bytes: int | None
    relative_threshold: float | None
    discarded_commutator_l2_fraction: float
    device_function_applied: bool


@dataclass(slots=True)
class ExperimentalConditions:
    """Explicit experimental parameters required for spectroscopy."""

    temperature: float
    pressure: float
    optical_length: float
    T2: float
    molecular_mass: float

    def __post_init__(self) -> None:
        for name in (
            "temperature",
            "pressure",
            "optical_length",
            "T2",
            "molecular_mass",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)

    @property
    def number_density(self) -> float:
        """数密度を計算 [m^-3]"""
        return self.pressure / (KB * self.temperature)

    @property
    def coherence_decay_rate(self) -> float:
        """コヒーレンス減衰率 [rad/s]"""
        return 1 / (self.T2 * 1e-12)


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
    dipole_matrix : DipoleMatrixBase
        双極子行列オブジェクト（SI単位）
    conditions : ExperimentalConditions
        Explicit experimental conditions
    axes : str, default 'xy'
        使用する双極子成分 ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'等)
    pol_int : np.ndarray, optional
        相互作用光の偏光ベクトル [Ex, Ey, Ez]
    pol_det : np.ndarray, optional
        検出光の偏光ベクトル（Noneの場合pol_intと同じ）

    Examples
    --------
    >>> basis = LinMolBasis(V_max=2, J_max=10, use_M=True, ...)
    >>> H0 = basis.generate_H0()
    >>> dipole = LinMolDipoleMatrix(basis=basis, mu0=1.0e-30)
    >>> calculator = AbsorbanceCalculator(
    ...     basis, H0, dipole, conditions, axes='xyz'
    ... )
    >>> absorbance = calculator.calculate(rho, wavenumber, method='loop')
    """

    def __init__(
        self,
        basis: BasisBase,
        hamiltonian: Hamiltonian,
        dipole_matrix: DipoleMatrixBase,
        conditions: ExperimentalConditions,
        axes: str = "xy",
        pol_int: np.ndarray | None = None,
        pol_det: np.ndarray | None = None,
        use_v_mask: bool = True,
    ):
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.dipole_matrix = dipole_matrix
        self.conditions = conditions

        # 軸の検証と設定
        self.axes = axes.lower()
        self._validate_axes()

        # 偏光ベクトルの設定（3次元）
        if pol_int is None:
            # デフォルト: x偏光
            pol_int = np.array([1.0, 0.0])
        else:
            pol_int = np.asarray(pol_int)

        self.pol_int = pol_int / np.linalg.norm(pol_int)
        self.pol_det = pol_det if pol_det is not None else self.pol_int.copy()

        self.pol_det = self.pol_det / np.linalg.norm(self.pol_det)

        self.use_v_mask = use_v_mask
        # 計算用の内部変数を初期化
        self._setup_matrices()
        self._prepared_2d = False
        self._last_calculation_report: SpectroscopyCalculationReport | None = None
        self._last_discarded_commutator_l2_fraction = 0.0

    def _validate_axes(self):
        """軸指定の検証"""
        valid_chars = set("xyz")
        if not all(c in valid_chars for c in self.axes):
            raise ValueError(
                f"Invalid axes '{self.axes}'. Must contain only 'x', 'y', 'z'."
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

        # 振動準位差マスクを作成
        if self.use_v_mask:
            self._create_v_mask()
        else:
            self.rho_mask = np.ones((self.N_level, self.N_level))

        # 遷移双極子行列を取得
        self._setup_dipole_matrices()

    def _create_v_mask(self):
        """振動準位差に基づくマスクを作成"""
        # BasisがV配列を持っている場合
        v_array = getattr(self.basis, "V_array", None)
        if v_array is not None:
            v_i = v_array.reshape(-1, 1)
            v_j = v_array.reshape(1, -1)
            # v差が1以内の要素のみを許可
            self.rho_mask = (np.abs(v_i - v_j) < 2).astype(float)
        else:
            # V配列がない場合は全要素を許可
            self.rho_mask = np.ones((self.N_level, self.N_level))

    def _setup_dipole_matrices(self):
        """双極子行列の設定（3次元対応）"""
        # 各軸の双極子成分を取得
        self.mu_components = {}
        mu_dict = {
            "x": self.dipole_matrix.get_mu_x_SI(),
            "y": self.dipole_matrix.get_mu_y_SI(),
            "z": self.dipole_matrix.get_mu_z_SI(),
        }
        self.mu_components[0] = (
            mu_dict[self.axes[0]]
            if self.axes[0] in mu_dict
            else np.zeros((self.N_level, self.N_level))
        )
        self.mu_components[1] = (
            mu_dict[self.axes[1]]
            if len(self.axes) > 1 and self.axes[1] in mu_dict
            else np.zeros((self.N_level, self.N_level))
        )
        # print("mu_components[0]", self.mu_components[0])
        # print("mu_components[1]", self.mu_components[1])
        # 偏光を考慮した双極子行列
        self.mu_int = (
            self.mu_components[0] * self.pol_int[0]
            + self.mu_components[1] * self.pol_int[1]
        )
        if not isinstance(self.mu_int, np.ndarray):
            self.mu_int = self.mu_int.toarray()
        # print(f"mu_int sample values: {self.mu_int[np.where(self.mu_int!=0)][:5]}")
        # 検出偏光を考慮した双極子行列
        self.mu_det = (
            self.mu_components[0] * self.pol_det[0]
            + self.mu_components[1] * self.pol_det[1]
        )
        if not isinstance(self.mu_det, np.ndarray):
            self.mu_det = self.mu_det.toarray()
        # print(f"mu_det sample values: {self.mu_det[np.where(self.mu_det!=0)][:5]}")

        # Detection support determines which response entries contribute.
        self.ind_nonzero = np.array(np.where(self.mu_det != 0))

    def prepare_2d_calculation(self, wavenumber: np.ndarray):
        """
        2D計算用の事前準備（高速化のため）

        Parameters
        ----------
        wavenumber : np.ndarray
            波数配列 [cm^-1]
        """
        omega = 2 * np.pi * C * 1e2 * wavenumber  # rad/s

        # 周波数配列を2D化
        omega_2d = omega.reshape(-1, 1)

        # 各遷移に対する分母を事前計算
        n_freq = len(omega)
        n_trans = self.ind_nonzero.shape[1]

        self._omega_2d = omega_2d
        self._one_over_denominator = np.zeros((n_freq, n_trans), dtype=np.complex128)

        for idx, trans in enumerate(self.ind_nonzero.T):
            i, j = tuple(trans)
            self._one_over_denominator[:, idx] = 1 / (
                1j * (omega + self.omega_vj_vpjp_mat[i, j])
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
        values = np.asarray(grid, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"{name} must contain at least two points")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        differences = np.diff(values)
        if np.any(differences == 0.0) or not (
            np.all(differences > 0.0) or np.all(differences < 0.0)
        ):
            raise ValueError(f"{name} must be strictly monotonic")
        reference = differences[0]
        tolerance = np.finfo(float).eps * max(1.0, np.max(np.abs(values))) * 16.0
        if not np.allclose(differences, reference, rtol=1.0e-12, atol=tolerance):
            raise ValueError(f"{name} must be uniformly spaced")
        return abs(float(reference))

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
        response_relevant = ~np.eye(self.N_level, dtype=bool)
        response_relevant &= self.mu_det.T != 0.0
        nonzero = response_relevant & (commutator != 0.0)

        if relative_threshold is None or not np.any(nonzero):
            self._last_discarded_commutator_l2_fraction = 0.0
            return np.where(nonzero)

        magnitudes = np.abs(commutator)
        scale = float(np.max(magnitudes[nonzero]))
        retained = nonzero & (magnitudes >= relative_threshold * scale)
        discarded = nonzero & ~retained
        total_norm = float(np.linalg.norm(commutator[nonzero]))
        discarded_norm = float(np.linalg.norm(commutator[discarded]))
        self._last_discarded_commutator_l2_fraction = (
            discarded_norm / total_norm if total_norm > 0.0 else 0.0
        )
        return np.where(retained)

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
        apply_doppler: bool = False,
        apply_device_function: bool = False,
        device_resolution: float | None = None,
        chunk_size: int | None = None,
        relative_threshold: float | None = None,
        memory_budget_bytes: int | None = None,
    ) -> np.ndarray:
        """Calculate an absorbance spectrum through an explicit method."""
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
            if apply_doppler:
                raise ValueError(
                    "auto with Doppler broadening is not available until all exact "
                    "methods share one characterized broadening kernel; choose an "
                    "explicit method"
                )
        elif memory_budget_bytes is not None:
            raise ValueError("memory_budget_bytes is applicable only to auto")

        if apply_device_function:
            if (
                device_resolution is None
                or not np.isfinite(device_resolution)
                or device_resolution <= 0.0
            ):
                raise ValueError(
                    "a finite positive device_resolution is required when "
                    "apply_device_function=True"
                )
        elif device_resolution is not None:
            raise ValueError(
                "device_resolution is applicable only when apply_device_function=True"
            )

        rho_array = np.asarray(rho, dtype=np.complex128)
        wavenumber_array = np.asarray(wavenumber, dtype=float)
        if apply_doppler or apply_device_function:
            self._uniform_grid_spacing(wavenumber_array, name="wavenumber")

        estimated_2d_bytes = self._estimate_2d_bytes(wavenumber_array)
        requested_method = method
        if method == "auto":
            executed_method = (
                "2d" if estimated_2d_bytes <= memory_budget_bytes else "chunked"
            )
        elif method == "approximate_sparse":
            executed_method = "chunked"
        else:
            executed_method = method

        self._last_discarded_commutator_l2_fraction = 0.0
        if executed_method == "chunked":
            spectrum = self._calculate_chunked(
                rho_array,
                wavenumber_array,
                chunk_size=chunk_size,
                apply_doppler=apply_doppler,
                relative_threshold=relative_threshold,
            )
        elif executed_method == "2d":
            spectrum = self._calculate_2d(
                rho_array,
                wavenumber_array,
                apply_doppler,
            )
        elif executed_method == "matrix":
            spectrum = self._calculate_matrix(
                rho_array,
                wavenumber_array,
                apply_doppler,
            )
        else:
            spectrum = self._calculate_loop(
                rho_array,
                wavenumber_array,
                apply_doppler,
            )

        if apply_device_function:
            spectrum = self.apply_device_function(
                spectrum,
                wavenumber_array,
                resolution=device_resolution,
            )

        self._last_calculation_report = SpectroscopyCalculationReport(
            requested_method=requested_method,
            executed_method=executed_method,
            estimated_2d_bytes=estimated_2d_bytes,
            memory_budget_bytes=memory_budget_bytes,
            relative_threshold=relative_threshold,
            discarded_commutator_l2_fraction=(
                self._last_discarded_commutator_l2_fraction
            ),
            device_function_applied=apply_device_function,
        )
        return spectrum

    def _calculate_2d(
        self, rho: np.ndarray, wavenumber: np.ndarray, apply_doppler: bool = False
    ) -> np.ndarray:
        """2D配列を使った高速計算"""
        # 事前準備がされていない、または波数が異なる場合は準備
        if not self._prepared_2d or not np.array_equal(
            wavenumber, self._prepared_wavenumber
        ):
            self.prepare_2d_calculation(wavenumber)

        # マスクを適用
        rho_masked = rho * self.rho_mask

        # コミュテータ [μ_int, ρ]
        rho_after_int = self.mu_int @ rho_masked - rho_masked @ self.mu_int

        # 強度因子を計算
        intensity_factors = np.zeros(self.ind_nonzero.shape[1], dtype=np.complex128)
        for idx, trans in enumerate(self.ind_nonzero.T):
            i, j = tuple(trans)
            intensity_factors[idx] = (
                -1j / H_DIRAC * self.mu_det[i, j] * rho_after_int[j, i]
            )

        # 2D演算で応答を計算
        resp_lin_per_mole_2d = self._one_over_denominator * intensity_factors
        resp_lin_per_mole = np.sum(resp_lin_per_mole_2d, axis=1)

        if apply_doppler:
            # ドップラー拡がりを適用
            resp_lin_per_mole = self._apply_doppler_broadening_full(
                wavenumber, resp_lin_per_mole
            )

        # 吸光度への変換
        omega = self._omega_2d[:, 0]
        return self._response_to_absorbance(omega, resp_lin_per_mole)

    def _calculate_matrix(
        self, rho: np.ndarray, wavenumber: np.ndarray, apply_doppler: bool = False
    ) -> np.ndarray:
        """行列演算による計算"""
        omega = 2 * np.pi * C * 1e2 * wavenumber  # rad/s

        # マスクを適用
        rho_masked = rho * self.rho_mask

        # コミュテータ [μ_int, ρ]
        rho_after_int = self.mu_int @ rho_masked - rho_masked @ self.mu_int

        # 各遷移に対する応答を計算
        responses = []
        for trans in self.ind_nonzero.T:
            i, j = tuple(trans)
            response = (
                -1j
                / H_DIRAC
                * self.mu_det[i, j]
                * rho_after_int[j, i]
                / (1j * (omega + self.omega_vj_vpjp_mat[i, j]))
            )

            if apply_doppler:
                omega_trans = float(np.real(self.omega_vj_vpjp_mat[i, j]))
                response = self._apply_doppler_broadening(omega, response, omega_trans)

            responses.append(response)

        # 全応答の和
        resp_lin_per_mole = np.sum(responses, axis=0)

        # 吸光度への変換
        return self._response_to_absorbance(omega, resp_lin_per_mole)

    def _calculate_loop(
        self, rho: np.ndarray, wavenumber: np.ndarray, apply_doppler: bool = False
    ) -> np.ndarray:
        """ループによる計算（メモリ効率重視）"""
        omega = 2 * np.pi * C * 1e2 * wavenumber

        rho_masked = rho * self.rho_mask
        rho_after_int = self.mu_int @ rho_masked - rho_masked @ self.mu_int

        resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)

        for trans in self.ind_nonzero.T:
            i, j = tuple(trans)
            response = (
                -1j
                / H_DIRAC
                * self.mu_det[i, j]
                * rho_after_int[j, i]
                / (1j * (omega + self.omega_vj_vpjp_mat[i, j]))
            )

            if apply_doppler:
                omega_trans = float(np.real(self.omega_vj_vpjp_mat[i, j]))
                response = self._apply_doppler_broadening(omega, response, omega_trans)

            resp_lin_per_mole += response

        return self._response_to_absorbance(omega, resp_lin_per_mole)

    def _calculate_chunked(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        chunk_size: int,
        apply_doppler: bool = False,
        relative_threshold: float | None = None,
    ) -> np.ndarray:
        """
        Memory-efficient chunked calculation for large systems
        """
        from scipy.sparse import csr_matrix

        # 密度行列にマスクを適用
        rho_masked = rho * self.rho_mask

        # 応答行列を計算（疎行列最適化）
        mu_int_sparse = csr_matrix(self.mu_int)
        rho_sparse = csr_matrix(rho_masked)

        # コミュテータ [mu_int, rho] を疎行列で計算
        rho1_sparse = mu_int_sparse @ rho_sparse - rho_sparse @ mu_int_sparse

        # Exact mode retains every response-relevant nonzero element.
        # Approximate mode applies an explicit scale-relative cutoff.
        rho1_dense = rho1_sparse.toarray()  # type: ignore
        i_indices, j_indices = self._response_entry_indices(
            rho1_dense,
            relative_threshold,
        )

        if len(i_indices) == 0:
            return np.zeros_like(wavenumber)

        # 周波数をチャンクに分割
        response = np.zeros(len(wavenumber), dtype=complex)

        for start_idx in range(0, len(wavenumber), chunk_size):
            end_idx = min(start_idx + chunk_size, len(wavenumber))
            omega_chunk = (
                2 * np.pi * C * wavenumber[start_idx:end_idx] * 100
            )  # cm^-1 to rad/s

            # チャンクごとに応答を計算
            response_chunk = np.zeros(len(omega_chunk), dtype=complex)

            for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
                if i != j:  # 非対角要素のみ
                    omega_ij = self.omega_vj_vpjp_mat[j, i]
                    mu_det_ij = self.mu_det[j, i]  # 検出双極子
                    rho1_ij = rho1_dense[i, j]

                    # 応答関数: -1 / (i*(omega + omega_ij))
                    denominator = 1j * (omega_chunk + omega_ij)
                    kernel = -1.0 / denominator

                    response_chunk += (1j / H_DIRAC) * mu_det_ij * rho1_ij * kernel

            response[start_idx:end_idx] = response_chunk

        # 吸光度に変換
        omega = 2 * np.pi * C * wavenumber * 100
        absorbance = self._response_to_absorbance(omega, response)

        if apply_doppler:
            absorbance = self._apply_doppler_broadening_full(wavenumber, absorbance)

        return absorbance

    def _response_to_absorbance(
        self, omega: np.ndarray, response: np.ndarray
    ) -> np.ndarray:
        """線形応答を吸光度に変換 [mOD]"""
        dens_num = self.conditions.number_density

        result = np.sqrt(1 + response / EPS * dens_num / 3)
        absorbance = (
            2 * self.conditions.optical_length * omega / C * result.imag  # type: ignore
        )

        # mODに変換
        absorbance *= np.log10(np.exp(1)) * 1000

        return absorbance

    @staticmethod
    def _filter_complex_gaussian(
        response: np.ndarray,
        sigma_pixels: float,
    ) -> np.ndarray:
        response_real = ndimage.gaussian_filter1d(
            response.real,
            sigma_pixels,
            mode="reflect",
        )
        response_imag = ndimage.gaussian_filter1d(
            response.imag,
            sigma_pixels,
            mode="reflect",
        )
        return response_real + 1j * response_imag

    def _apply_doppler_broadening(
        self,
        omega: np.ndarray,
        response: np.ndarray,
        omega0: float,
    ) -> np.ndarray:
        """Apply transition-specific Doppler broadening on its actual grid."""
        if omega0 == 0.0:
            return response
        spacing = self._uniform_grid_spacing(omega, name="angular-frequency grid")
        sigma_doppler = abs(omega0) * np.sqrt(
            KB * self.conditions.temperature / (self.conditions.molecular_mass * C**2)
        )
        return self._filter_complex_gaussian(
            response,
            sigma_doppler / spacing,
        )

    def _apply_doppler_broadening_full(
        self,
        wavenumber: np.ndarray,
        response: np.ndarray,
    ) -> np.ndarray:
        """Apply aggregate Doppler broadening resolved on the wavenumber grid."""
        spacing = self._uniform_grid_spacing(wavenumber, name="wavenumber")
        energy_diffs = []
        for i in range(self.N_level):
            for j in range(i + 1, self.N_level):
                diff = abs(self.energy_array[i] - self.energy_array[j])
                if diff > 0.0:
                    energy_diffs.append(diff)

        if not energy_diffs:
            return response

        mean_omega = float(np.mean(energy_diffs)) / H_DIRAC
        sigma_doppler_wn = (
            mean_omega
            / (2 * np.pi * C * 1e2)
            * np.sqrt(
                KB
                * self.conditions.temperature
                / (self.conditions.molecular_mass * C**2)
            )
        )
        return self._filter_complex_gaussian(
            response,
            sigma_doppler_wn / spacing,
        )

    def calculate_radiation_spectrum(
        self, rho: np.ndarray, wavenumber: np.ndarray
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

        Returns
        -------
        np.ndarray
            放射スペクトル [mOD]
        """
        omega = 2 * np.pi * C * 1e2 * wavenumber

        rho_masked = rho * self.rho_mask
        resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)

        for trans in self.ind_nonzero.T:
            i, j = tuple(trans)
            # 放射の場合は順序が逆
            resp_lin_per_mole += -(
                self.mu_det[j, i]
                * rho_masked[i, j]
                / (1j * (omega + self.omega_vj_vpjp_mat[i, j]))
            )

        return self._response_to_absorbance(omega, resp_lin_per_mole)

    def calculate_pfid_spectrum(
        self, rho: np.ndarray, wavenumber: np.ndarray
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

        Returns
        -------
        np.ndarray
            PFIDスペクトル [mOD]
        """
        # PFIDは放射スペクトルと同じ計算
        return self.calculate_radiation_spectrum(rho, wavenumber)

    def apply_device_function(
        self,
        spectrum: np.ndarray,
        wavenumber: np.ndarray,
        resolution: float = 1.0,
        function_type: Literal["sinc", "sinc2", "gaussian"] = "sinc2",
    ) -> np.ndarray:
        """
        装置関数を適用

        Parameters
        ----------
        spectrum : np.ndarray
            スペクトル
        wavenumber : np.ndarray
            波数配列 [cm^-1]
        resolution : float
            分解能 [cm^-1]
        function_type : {'sinc', 'sinc2', 'gaussian'}
            装置関数のタイプ

        Returns
        -------
        np.ndarray
            装置関数適用後のスペクトル
        """
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be finite and positive")
        if function_type not in {"sinc", "sinc2", "gaussian"}:
            raise ValueError(f"unknown device function: {function_type}")

        dw = self._uniform_grid_spacing(wavenumber, name="wavenumber")

        if function_type == "gaussian":
            # ガウシアン装置関数
            sigma_pixels = resolution / (2 * np.sqrt(2 * np.log(2))) / dw
            return ndimage.gaussian_filter1d(spectrum, sigma_pixels, mode="reflect")

        else:
            # Sinc または Sinc^2 装置関数
            # FFTベースの畳み込み
            n = len(wavenumber)
            x_device = np.arange(-n // 2, n // 2) * dw

            if function_type == "sinc":
                device_func = np.sinc(2 * x_device / resolution)
            else:  # sinc2
                device_func = np.sinc(2 * x_device / resolution) ** 2

            device_func /= np.sum(device_func)  # 正規化

            # 畳み込み
            return np.convolve(spectrum, device_func, mode="same")


# ヘルパー関数
def create_calculator_from_params(
    basis: BasisBase,
    hamiltonian: Hamiltonian,
    dipole_matrix: DipoleMatrixBase,
    *,
    temperature: float,
    pressure: float,
    optical_length: float,
    T2: float,
    molecular_mass: float,
    axes: str = "xy",
    pol_int: np.ndarray | None = None,
    pol_det: np.ndarray | None = None,
) -> AbsorbanceCalculator:
    """
    パラメータから計算機を作成するヘルパー関数

    Parameters
    ----------
    basis : BasisBase
        量子基底
    hamiltonian : Hamiltonian
        ハミルトニアン
    dipole_matrix : DipoleMatrixBase
        双極子行列
    temperature : float
        温度 [K]
    pressure : float
        圧力 [Pa]
    optical_length : float
        光路長 [m]
    T2 : float
        コヒーレンス緩和時間 [ps]
    molecular_mass : float
        分子質量 [kg]
    axes : str
        使用する軸
    pol_int, pol_det : np.ndarray, optional
        偏光ベクトル

    Returns
    -------
    AbsorbanceCalculator
        初期化された計算機オブジェクト
    """
    conditions = ExperimentalConditions(
        temperature=temperature,
        pressure=pressure,
        optical_length=optical_length,
        T2=T2,
        molecular_mass=molecular_mass,
    )

    return AbsorbanceCalculator(
        basis=basis,
        hamiltonian=hamiltonian,
        dipole_matrix=dipole_matrix,
        conditions=conditions,
        axes=axes,
        pol_int=pol_int,
        pol_det=pol_det,
    )
