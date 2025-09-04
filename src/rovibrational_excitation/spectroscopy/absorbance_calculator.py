#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吸光度スペクトル計算モジュール

密度行列から吸光度スペクトルを計算するためのクラスと関数を提供。
@core/の標準オブジェクト（Basis, Hamiltonian, DipoleMatrix）と統合。
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Union
import numpy as np
from scipy import ndimage
from scipy.special import wofz

from rovibrational_excitation.core.basis import BasisBase
from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
from rovibrational_excitation.dipole.base import DipoleMatrixBase


# 物理定数
H_DIRAC = 1.055e-34  # ディラック定数 [J*s]
EE = 1.601e-19  # 素電荷 [C]
C = 2.998e8  # 光速 [m/s]
EPS = 8.854e-12  # 真空の誘電率 [F/m]
KB = 1.380649e-23  # ボルツマン定数 [J/K]


@dataclass
class ExperimentalConditions:
    """実験条件のパラメータ"""
    temperature: float = 300  # K
    pressure: float = 3e4  # Pa
    optical_length: float = 1e-3  # m
    T2: float = 500  # ps (coherence relaxation time)
    molecular_mass: float = 44e-3/6.023e23  # kg (CO2 default)
    
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
    conditions : ExperimentalConditions, optional
        実験条件
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
    >>> calculator = AbsorbanceCalculator(basis, H0, dipole, axes='xyz')
    >>> absorbance = calculator.calculate(rho, wavenumber)
    """
    
    def __init__(
        self,
        basis: BasisBase,
        hamiltonian: Hamiltonian,
        dipole_matrix: DipoleMatrixBase,
        conditions: Optional[ExperimentalConditions] = None,
        axes: str = 'xy',
        pol_int: Optional[np.ndarray] = None,
        pol_det: Optional[np.ndarray] = None,
    ):
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.dipole_matrix = dipole_matrix
        self.conditions = conditions or ExperimentalConditions()
        
        # 軸の検証と設定
        self.axes = axes.lower()
        self._validate_axes()
        
        # 偏光ベクトルの設定（3次元）
        if pol_int is None:
            # デフォルト: x偏光
            pol_int = np.array([1.0, 0.0, 0.0])
        else:
            pol_int = np.asarray(pol_int)
            if pol_int.shape == (2,):
                # 2次元の場合は3次元に拡張
                pol_int = np.array([pol_int[0], pol_int[1], 0.0])
        
        self.pol_int = pol_int / np.linalg.norm(pol_int)
        self.pol_det = pol_det if pol_det is not None else self.pol_int.copy()
        
        if self.pol_det.shape == (2,):
            self.pol_det = np.array([self.pol_det[0], self.pol_det[1], 0.0])
        self.pol_det = self.pol_det / np.linalg.norm(self.pol_det)
        
        # 計算用の内部変数を初期化
        self._setup_matrices()
        self._prepared_2d = False
    
    def _validate_axes(self):
        """軸指定の検証"""
        valid_chars = set('xyz')
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
            (energy_vstack - energy_vstack.T) / H_DIRAC - 1j * gamma_coh
        )
        
        # 振動準位差マスクを作成
        self._create_v_mask()
        
        # 遷移双極子行列を取得
        self._setup_dipole_matrices()
    
    def _create_v_mask(self):
        """振動準位差に基づくマスクを作成"""
        # BasisがV配列を持っている場合
        v_array = getattr(self.basis, 'V_array', None)
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
        
        if 'x' in self.axes:
            self.mu_components['x'] = self.dipole_matrix.get_mu_x_SI()
        else:
            self.mu_components['x'] = np.zeros((self.N_level, self.N_level))
            
        if 'y' in self.axes:
            self.mu_components['y'] = self.dipole_matrix.get_mu_y_SI()
        else:
            self.mu_components['y'] = np.zeros((self.N_level, self.N_level))
            
        if 'z' in self.axes:
            # z成分をサポート
            if hasattr(self.dipole_matrix, 'get_mu_z_SI'):
                self.mu_components['z'] = self.dipole_matrix.get_mu_z_SI()
            else:
                # z成分がない場合は警告してゼロ行列
                import warnings
                warnings.warn(
                    f"Dipole matrix {type(self.dipole_matrix).__name__} "
                    "does not support z-component. Using zeros.",
                    UserWarning
                )
                self.mu_components['z'] = np.zeros((self.N_level, self.N_level))
        else:
            self.mu_components['z'] = np.zeros((self.N_level, self.N_level))
        
        # 偏光を考慮した双極子行列
        self.mu_int = (
            self.mu_components['x'] * self.pol_int[0] +
            self.mu_components['y'] * self.pol_int[1] +
            self.mu_components['z'] * self.pol_int[2]
        )
        
        self.mu_det = (
            self.mu_components['x'] * self.pol_det[0] +
            self.mu_components['y'] * self.pol_det[1] +
            self.mu_components['z'] * self.pol_det[2]
        )
        
        # 非ゼロ遷移のインデックス
        self.ind_nonzero = np.array(np.where(self.mu_int != 0))
    
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
        self._prepared_wavenumber = wavenumber
    
    def calculate(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        method: Literal['matrix', 'loop', '2d', 'chunked', 'optimized'] = 'optimized',
        apply_doppler: bool = False,
        apply_device_function: bool = False,
        device_resolution: float = 1.0,  # cm^-1
        chunk_size: int = 1000,  # For chunked methods
        sparse_threshold: float = 1e-12  # For sparse optimization
    ) -> np.ndarray:
        """
        吸光度スペクトルを計算
        
        Parameters
        ----------
        rho : np.ndarray
            密度行列
        wavenumber : np.ndarray
            波数配列 [cm^-1]
        method : {'matrix', 'loop', '2d', 'chunked', 'optimized'}
            計算方法
            - 'matrix': 行列演算（メモリ使用量大、高速）
            - 'loop': ループ演算（メモリ効率的、低速）
            - '2d': 2D配列を使った高速計算
            - 'chunked': チャンク化による省メモリ計算
            - 'optimized': 自動最適化（推奨）
        apply_doppler : bool
            ドップラー拡がりを適用するか
        apply_device_function : bool
            装置関数（sinc関数）を適用するか
        device_resolution : float
            装置分解能 [cm^-1]
        chunk_size : int
            チャンク化時のサイズ
        sparse_threshold : float
            疎行列の閾値
            
        Returns
        -------
        np.ndarray
            吸光度スペクトル [mOD]
        """
        if method == 'optimized':
            return self._calculate_smart_optimized(rho, wavenumber, chunk_size, sparse_threshold, apply_doppler)
        elif method == 'chunked':
            return self._calculate_chunked(rho, wavenumber, chunk_size, apply_doppler)
        elif method == '2d':
            return self._calculate_2d(rho, wavenumber, apply_doppler)
        elif method == 'matrix':
            return self._calculate_matrix(rho, wavenumber, apply_doppler)
        elif method == 'loop':
            return self._calculate_loop(rho, wavenumber, apply_doppler)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calculate_2d(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        apply_doppler: bool = False
    ) -> np.ndarray:
        """2D配列を使った高速計算"""
        # 事前準備がされていない、または波数が異なる場合は準備
        if not self._prepared_2d or not np.array_equal(wavenumber, self._prepared_wavenumber):
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
                -1j / H_DIRAC *
                self.mu_det[i, j] * rho_after_int[j, i]
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
        self, 
        rho: np.ndarray, 
        wavenumber: np.ndarray,
        apply_doppler: bool = False
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
                -1j / H_DIRAC * 
                self.mu_det[i, j] * rho_after_int[j, i] /
                (1j * (omega + self.omega_vj_vpjp_mat[i, j]))
            )
            
            if apply_doppler:
                omega_trans = float(np.real(self.omega_vj_vpjp_mat[i, j]))
                response = self._apply_doppler_broadening(
                    omega, response, omega_trans
                )
            
            responses.append(response)
        
        # 全応答の和
        resp_lin_per_mole = np.sum(responses, axis=0)
        
        # 吸光度への変換
        return self._response_to_absorbance(omega, resp_lin_per_mole)
    
    def _calculate_loop(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        apply_doppler: bool = False
    ) -> np.ndarray:
        """ループによる計算（メモリ効率重視）"""
        omega = 2 * np.pi * C * 1e2 * wavenumber
        
        rho_masked = rho * self.rho_mask
        rho_after_int = self.mu_int @ rho_masked - rho_masked @ self.mu_int
        
        resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
        
        for trans in self.ind_nonzero.T:
            i, j = tuple(trans)
            response = (
                -1j / H_DIRAC *
                self.mu_det[i, j] * rho_after_int[j, i] /
                (1j * (omega + self.omega_vj_vpjp_mat[i, j]))
            )
            
            if apply_doppler:
                omega_trans = float(np.real(self.omega_vj_vpjp_mat[i, j]))
                response = self._apply_doppler_broadening(
                    omega, response, omega_trans
                )
            
            resp_lin_per_mole += response
        
        return self._response_to_absorbance(omega, resp_lin_per_mole)
    
    def _calculate_optimized(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        chunk_size: int = 1000,
        sparse_threshold: float = 1e-12,
        apply_doppler: bool = False
    ) -> np.ndarray:
        """
        Optimized calculation with automatic method selection based on memory constraints
        """
        # メモリ使用量を推定
        N_level = self.N_level
        N_freq = len(wavenumber)
        
        # 基本のメモリ推定（複素数配列のサイズ）
        memory_matrix = N_level * N_level * 16  # bytes for complex128
        memory_frequency = N_freq * N_level * N_level * 16  # for full response calculation
        
        # メモリ制限（4GB以下なら matrix method, それ以上なら chunked）
        memory_limit = 4 * 1024**3  # 4GB
        
        if memory_frequency < memory_limit and N_level < 1000:
            # 小さなシステム: 高速な2D method
            return self._calculate_2d(rho, wavenumber, apply_doppler)
        else:
            # 大きなシステム: メモリ効率的なchunked method
            return self._calculate_chunked(rho, wavenumber, chunk_size, apply_doppler)
    
    def _calculate_chunked(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        chunk_size: int = 1000,
        apply_doppler: bool = False
    ) -> np.ndarray:
        """
        Memory-efficient chunked calculation for large systems
        """
        from scipy.sparse import csr_matrix
        
        # 密度行列にマスクを適用
        rho_masked = rho * self.rho_mask
        
        # 応答行列を計算（疎行列最適化）
        mu_int_sparse = csr_matrix(self.mu_int)
        mu_det_sparse = csr_matrix(self.mu_det)
        rho_sparse = csr_matrix(rho_masked)
        
        # コミュテータ [mu_int, rho] を疎行列で計算
        rho1_sparse = mu_int_sparse @ rho_sparse - rho_sparse @ mu_int_sparse
        
        # 非ゼロ要素のインデックスを特定
        rho1_dense = rho1_sparse.toarray()  # type: ignore
        nonzero_mask = np.abs(rho1_dense) > 1e-15
        i_indices, j_indices = np.where(nonzero_mask)
        
        if len(i_indices) == 0:
            return np.zeros_like(wavenumber)
        
        # 周波数をチャンクに分割
        response = np.zeros(len(wavenumber), dtype=complex)
        
        for start_idx in range(0, len(wavenumber), chunk_size):
            end_idx = min(start_idx + chunk_size, len(wavenumber))
            omega_chunk = 2 * np.pi * C * wavenumber[start_idx:end_idx] * 100  # cm^-1 to rad/s
            
            # チャンクごとに応答を計算
            response_chunk = np.zeros(len(omega_chunk), dtype=complex)
            
            for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
                if i != j:  # 非対角要素のみ
                    omega_ij = self.omega_vj_vpjp_mat[i, j]
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
    
    def _calculate_smart_optimized(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        chunk_size: int = 1000,
        sparse_threshold: float = 1e-12,
        apply_doppler: bool = False
    ) -> np.ndarray:
        """
        Smart memory-optimized calculation with automatic method selection
        """
        N_level = self.N_level
        N_freq = len(wavenumber)
        
        # メモリ使用量を推定 (最悪ケース)
        memory_matrix_gb = (N_level * N_level * 16) / (1024**3)  # 密度行列等
        memory_frequency_gb = (N_freq * 8) / (1024**3)  # 周波数配列
        
        if N_level < 500:
            # 小さなシステム: 高速な2D method
            print(f"Using fast 2D method (N={N_level}, matrix memory: {memory_matrix_gb:.2f} GB)")
            return self._calculate_2d(rho, wavenumber, apply_doppler)
        elif N_level < 1500:
            # 中程度のシステム: loop method
            print(f"Using memory-efficient loop method (N={N_level}, matrix memory: {memory_matrix_gb:.2f} GB)")
            return self._calculate_loop(rho, wavenumber, apply_doppler)
        else:
            # 大きなシステム: 専用チャンク化手法
            print(f"Using ultra-efficient chunked method (N={N_level}, matrix memory: {memory_matrix_gb:.2f} GB)")
            return self._calculate_ultra_chunked(rho, wavenumber, chunk_size, apply_doppler)
    
    def _calculate_ultra_chunked(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray,
        chunk_size: int = 500,
        apply_doppler: bool = False
    ) -> np.ndarray:
        """
        Ultra-memory-efficient calculation for very large systems
        """
        # 周波数のチャンクサイズを動的に調整
        N_level = self.N_level
        if N_level > 2000:
            freq_chunk_size = min(chunk_size // 2, 200)
        elif N_level > 1000:
            freq_chunk_size = min(chunk_size, 500)
        else:
            freq_chunk_size = chunk_size
        
        print(f"Processing {len(wavenumber)} frequencies in chunks of {freq_chunk_size}")
        
        # 密度行列にマスクを適用
        rho_masked = rho * self.rho_mask
        
        # コミュテータを事前計算（疎行列化）
        from scipy.sparse import csr_matrix
        mu_int_sparse = csr_matrix(self.mu_int)
        rho_sparse = csr_matrix(rho_masked)
        
        # [mu_int, rho] = mu_int * rho - rho * mu_int
        rho1_sparse = mu_int_sparse @ rho_sparse - rho_sparse @ mu_int_sparse
        rho1_array = rho1_sparse.toarray()  # type: ignore
        
        # 非ゼロ要素のインデックスを取得
        nonzero_mask = np.abs(rho1_array) > 1e-15
        i_indices, j_indices = np.where(nonzero_mask)
        
        if len(i_indices) == 0:
            print("Warning: No non-zero transitions found!")
            return np.zeros_like(wavenumber)
        
        print(f"Found {len(i_indices)} non-zero transitions out of {N_level**2} possible")
        
        # デバッグ: いくつかの要素を確認
        print(f"Sample rho1 values: {rho1_array[i_indices[:5], j_indices[:5]]}")
        print(f"Sample mu_det values: {self.mu_det[j_indices[:5], i_indices[:5]]}")
        print(f"Sample omega_ij values: {self.omega_vj_vpjp_mat[i_indices[:5], j_indices[:5]]}")
        
        # 結果配列
        response_total = np.zeros(len(wavenumber), dtype=complex)
        
        # 周波数をチャンクに分割して処理
        for start_idx in range(0, len(wavenumber), freq_chunk_size):
            end_idx = min(start_idx + freq_chunk_size, len(wavenumber))
            wn_chunk = wavenumber[start_idx:end_idx]
            omega_chunk = 2 * np.pi * C * wn_chunk * 100  # cm^-1 to rad/s
            
            response_chunk = np.zeros(len(omega_chunk), dtype=complex)
            
            # 遷移をバッチ処理
            batch_size = min(100, len(i_indices))
            for batch_start in range(0, len(i_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(i_indices))
                
                for idx in range(batch_start, batch_end):
                    i, j = i_indices[idx], j_indices[idx]
                    if i != j:  # 非対角要素のみ
                        omega_ij = self.omega_vj_vpjp_mat[i, j]
                        mu_det_ij = self.mu_det[j, i]
                        rho1_ij = rho1_array[i, j]
                        
                        # 応答関数
                        denominator = 1j * (omega_chunk + omega_ij)
                        kernel = -1.0 / denominator
                        
                        response_chunk += (1j / H_DIRAC) * mu_det_ij * rho1_ij * kernel
            
            response_total[start_idx:end_idx] = response_chunk
            
            # 進捗表示
            progress = (end_idx / len(wavenumber)) * 100
            if start_idx % (freq_chunk_size * 5) == 0:
                print(f"  Progress: {progress:.1f}%")
        
        # 吸光度に変換
        omega = 2 * np.pi * C * wavenumber * 100
        absorbance = self._response_to_absorbance(omega, response_total)
        
        if apply_doppler:
            absorbance = self._apply_doppler_broadening_full(wavenumber, absorbance)
        
        return absorbance
    
    def _response_to_absorbance(
        self, 
        omega: np.ndarray, 
        response: np.ndarray
    ) -> np.ndarray:
        """線形応答を吸光度に変換 [mOD]"""
        dens_num = self.conditions.number_density
        
        result = np.sqrt(1 + response / EPS * dens_num / 3)
        absorbance = (
            2 * self.conditions.optical_length * omega / C *
            result.imag  # type: ignore
        )
        
        # mODに変換
        absorbance *= np.log10(np.exp(1)) * 1000
        
        return absorbance
    
    def _apply_doppler_broadening(
        self,
        omega: np.ndarray,
        response: np.ndarray,
        omega0: float
    ) -> np.ndarray:
        """
        単一遷移にドップラー拡がりを適用
        
        Parameters
        ----------
        omega : np.ndarray
            角周波数配列 [rad/s]
        response : np.ndarray
            応答関数
        omega0 : float
            遷移の中心角周波数 [rad/s]
        """
        if omega0 <= 0:
            return response
        
        # ドップラー幅（標準偏差）
        sigma_doppler = omega0 * np.sqrt(
            KB * self.conditions.temperature / 
            (self.conditions.molecular_mass * C**2)
        )
        
        if sigma_doppler < 1e-10:  # 幅が小さすぎる場合はスキップ
            return response
        
        # 周波数空間でのガウシアン畳み込み
        # 簡略化のため、ここでは単純なガウシアン乗算を行う
        # 実際にはFFTを使った畳み込みが望ましい
        doppler_profile = np.exp(-(omega - omega0)**2 / (2 * sigma_doppler**2))
        doppler_profile /= np.sum(doppler_profile)  # 正規化
        
        # 畳み込み
        return np.convolve(response, doppler_profile, mode='same')
    
    def _apply_doppler_broadening_full(
        self,
        wavenumber: np.ndarray,
        response: np.ndarray
    ) -> np.ndarray:
        """
        全体の応答にドップラー拡がりを適用（より正確な実装）
        """
        # 波数をrad/sに変換
        omega = 2 * np.pi * C * 1e2 * wavenumber
        
        # 平均遷移エネルギーを推定
        energy_diffs = []
        for i in range(self.N_level):
            for j in range(i+1, self.N_level):
                diff = abs(self.energy_array[i] - self.energy_array[j])
                if diff > 0:
                    energy_diffs.append(diff)
        
        if not energy_diffs:
            return response
        
        mean_omega = np.mean(energy_diffs) / H_DIRAC
        
        # ドップラー幅
        sigma_doppler_wn = mean_omega / (2 * np.pi * C * 1e2) * np.sqrt(
            KB * self.conditions.temperature / 
            (self.conditions.molecular_mass * C**2)
        )
        
        # ガウシアンフィルタを適用
        if sigma_doppler_wn > 0.01:  # 0.01 cm^-1以上の場合のみ
            # 波数空間でのピクセル数に変換
            dw = wavenumber[1] - wavenumber[0] if len(wavenumber) > 1 else 1.0
            sigma_pixels = sigma_doppler_wn / dw
            
            # scipy.ndimage.gaussian_filter1dを使用
            response_real = ndimage.gaussian_filter1d(
                response.real, sigma_pixels, mode='reflect'
            )
            response_imag = ndimage.gaussian_filter1d(
                response.imag, sigma_pixels, mode='reflect'
            )
            response = response_real + 1j * response_imag
        
        return response
    
    def calculate_radiation_spectrum(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray
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
                self.mu_det[j, i] * rho_masked[i, j] /
                (1j * (omega + self.omega_vj_vpjp_mat[i, j]))
            )
        
        return self._response_to_absorbance(omega, resp_lin_per_mole)
    
    def calculate_pfid_spectrum(
        self,
        rho: np.ndarray,
        wavenumber: np.ndarray
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
        function_type: Literal['sinc', 'sinc2', 'gaussian'] = 'sinc2'
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
        if resolution <= 0:
            return spectrum
        
        dw = wavenumber[1] - wavenumber[0] if len(wavenumber) > 1 else 1.0
        
        if function_type == 'gaussian':
            # ガウシアン装置関数
            sigma_pixels = resolution / (2 * np.sqrt(2 * np.log(2))) / dw
            return ndimage.gaussian_filter1d(spectrum, sigma_pixels, mode='reflect')
            
        else:
            # Sinc または Sinc^2 装置関数
            # FFTベースの畳み込み
            n = len(wavenumber)
            x_device = np.arange(-n//2, n//2) * dw
            
            if function_type == 'sinc':
                device_func = np.sinc(2 * x_device / resolution)
            else:  # sinc2
                device_func = np.sinc(2 * x_device / resolution) ** 2
            
            device_func /= np.sum(device_func)  # 正規化
            
            # 畳み込み
            return np.convolve(spectrum, device_func, mode='same')


# ヘルパー関数
def create_calculator_from_params(
    basis: BasisBase,
    hamiltonian: Hamiltonian,
    dipole_matrix: DipoleMatrixBase,
    temperature: float = 300,
    pressure: float = 3e4,
    optical_length: float = 1e-3,
    T2: float = 500,
    molecular_mass: Optional[float] = None,
    axes: str = 'xy',
    pol_int: Optional[np.ndarray] = None,
    pol_det: Optional[np.ndarray] = None
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
    molecular_mass : float, optional
        分子質量 [kg]（Noneの場合CO2のデフォルト値）
    axes : str
        使用する軸
    pol_int, pol_det : np.ndarray, optional
        偏光ベクトル
        
    Returns
    -------
    AbsorbanceCalculator
        初期化された計算機オブジェクト
    """
    # 分子質量の自動推定（簡易的）
    if molecular_mass is None:
        # 基底の種類から推測
        basis_name = type(basis).__name__
        if 'CO2' in basis_name or 'co2' in basis_name.lower():
            molecular_mass = 44e-3 / 6.023e23  # CO2
        elif 'CO' in basis_name or 'co' in basis_name.lower():
            molecular_mass = 28e-3 / 6.023e23  # CO
        else:
            molecular_mass = 44e-3 / 6.023e23  # デフォルト
    
    conditions = ExperimentalConditions(
        temperature=temperature,
        pressure=pressure,
        optical_length=optical_length,
        T2=T2,
        molecular_mass=molecular_mass
    )
    
    return AbsorbanceCalculator(
        basis=basis,
        hamiltonian=hamiltonian,
        dipole_matrix=dipole_matrix,
        conditions=conditions,
        axes=axes,
        pol_int=pol_int,
        pol_det=pol_det
    )
