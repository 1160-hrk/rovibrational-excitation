"""
Hamiltonian class with unit preservation.

This module focuses on physical unit conversions (J ↔ rad/fs) only.
For input unit conversions, use the units.py module.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class Hamiltonian:
    """
    ハミルトニアン行列と単位情報を保持するクラス
    
    物理的なハミルトニアンの値と単位を一緒に管理し、
    J ↔ rad/fs の厳密な物理変換のみを提供します。
    
    Notes
    -----
    多様な入力単位（cm⁻¹, THz, D など）の変換は units.py モジュールを使用してください。
    このクラスは物理計算レイヤーでの厳密な単位管理を担当します。
    """
    
    # Planck constant for physical unit conversions only
    _HBAR = 6.62607015e-034 / (2 * np.pi)  # J⋅s
    
    # Physical constants for input unit conversions
    _C = 2.99792458e10  # speed of light [cm/s]
    _H = 6.62607015e-34  # Planck constant [J·s]
    _E = 1.602176634e-19  # elementary charge [C]
    
    # Unit conversion factors for input units
    _FREQUENCY_CONVERSIONS = {
        # Target: rad/fs
        "rad/fs": 1.0,
        "THz": 2 * np.pi * 1e-3,  # THz → rad/fs
        "GHz": 2 * np.pi * 1e-6,  # GHz → rad/fs  
        "cm^-1": 2 * np.pi * _C * 1e-15,  # cm⁻¹ → rad/fs
        "cm-1": 2 * np.pi * _C * 1e-15,   # alternative notation
        "wavenumber": 2 * np.pi * _C * 1e-15,  # alias
        "PHz": 2 * np.pi,  # PHz → rad/fs
        "Hz": 2 * np.pi * 1e-15,  # Hz → rad/fs
        "rad/s": 1e-15,  # rad/s → rad/fs
    }
    
    _ENERGY_CONVERSIONS = {
        # Target: J
        "J": 1.0,
        "eV": _E,  # eV → J
        "meV": _E * 1e-3,  # meV → J
        "μJ": 1e-6,  # μJ → J
        "uJ": 1e-6,  # μJ → J (alternative notation)
        "mJ": 1e-3,  # mJ → J
        "nJ": 1e-9,  # nJ → J
        "pJ": 1e-12,  # pJ → J
        "cm^-1": _H * _C,  # cm⁻¹ → J
        "cm-1": _H * _C,
        "wavenumber": _H * _C,
    }
    
    def __init__(
        self, 
        matrix: np.ndarray, 
        units: Literal["J", "rad/fs"] = "J",
        basis_info: dict | None = None
    ):
        """
        Parameters
        ----------
        matrix : np.ndarray
            ハミルトニアン行列
        units : {"J", "rad/fs"}
            行列の単位
        basis_info : dict, optional
            基底に関する情報（デバッグ用）
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("matrix must be numpy ndarray")
        
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square 2D array")
            
        if units not in ["J", "rad/fs"]:
            raise ValueError("units must be 'J' or 'rad/fs'")
        
        self._matrix = matrix.copy()
        self._units: Literal["J", "rad/fs"] = units
        self._basis_info = basis_info or {}
    
    @classmethod
    def from_input_units(
        cls,
        matrix: np.ndarray,
        input_units: str,
        target_units: str = "J",
        basis_info: dict | None = None
    ) -> "Hamiltonian":
        """
        Create a Hamiltonian object from a matrix with input units.
        
        This method replaces the units.py create_hamiltonian_from_input_units function.
        
        Parameters
        ----------
        matrix : np.ndarray
            Hamiltonian matrix in input units
        input_units : str
            Units of the input matrix (e.g., "cm^-1", "THz", "eV")
        target_units : {"J", "rad/fs"}, optional
            Target units for the Hamiltonian object
        basis_info : dict, optional
            Basis information for debugging
            
        Returns
        -------
        Hamiltonian
            Hamiltonian object with proper unit management
            
        Examples
        --------
        >>> # Create Hamiltonian from cm⁻¹ matrix
        >>> h_matrix = np.diag([0, 2350])  # CO2 ν3 mode
        >>> h = Hamiltonian.from_input_units(h_matrix, "cm^-1", "J")
        
        >>> # Create from eV matrix  
        >>> h_matrix = np.diag([0, 1.5])  # 1.5 eV gap
        >>> h = Hamiltonian.from_input_units(h_matrix, "eV", "rad/fs")
        """
        # Convert input units to standardized internal units
        if input_units in cls._FREQUENCY_CONVERSIONS:
            # Frequency units → rad/fs (internal standard)
            matrix_rad_fs = matrix * cls._FREQUENCY_CONVERSIONS[input_units]
            hamiltonian = cls(matrix_rad_fs, "rad/fs", basis_info)
        elif input_units in cls._ENERGY_CONVERSIONS:
            # Energy units → J (internal standard)
            matrix_J = matrix * cls._ENERGY_CONVERSIONS[input_units]
            hamiltonian = cls(matrix_J, "J", basis_info)
        else:
            raise ValueError(f"Unsupported Hamiltonian units: {input_units}. "
                            f"Use frequency units {list(cls._FREQUENCY_CONVERSIONS.keys())} "
                            f"or energy units {list(cls._ENERGY_CONVERSIONS.keys())}")
        
        # Convert to target units if needed
        if target_units == "J":
            return hamiltonian.to_energy_units() if hamiltonian.units == "rad/fs" else hamiltonian
        elif target_units == "rad/fs":
            return hamiltonian.to_frequency_units() if hamiltonian.units == "J" else hamiltonian
        else:
            raise ValueError(f"Target units must be 'J' or 'rad/fs', got {target_units}")
    
    @property
    def matrix(self) -> np.ndarray:
        """ハミルトニアン行列"""
        return self._matrix.copy()
    
    @property
    def units(self) -> Literal["J", "rad/fs"]:
        """単位"""
        return self._units
    
    @property
    def shape(self) -> tuple[int, int]:
        """行列の形状"""
        return self._matrix.shape
    
    @property
    def size(self) -> int:
        """行列のサイズ（次元数）"""
        return self._matrix.shape[0]
    
    @property
    def eigenvalues(self) -> np.ndarray:
        """固有値（対角成分、現在の単位）"""
        if self.is_diagonal():
            return np.diag(self._matrix)
        else:
            return np.linalg.eigvals(self._matrix)
    
    def is_diagonal(self) -> bool:
        """対角行列かどうか"""
        return np.allclose(self._matrix, np.diag(np.diag(self._matrix)))
    
    def to_energy_units(self) -> Hamiltonian:
        """エネルギー単位（J）に変換"""
        if self._units == "J":
            return Hamiltonian(self._matrix, "J", self._basis_info)
        elif self._units == "rad/fs":
            # rad/fs → J: E = ℏω, rad/fs → rad/s → J
            matrix_J = self._matrix * self._HBAR / 1e-15
            return Hamiltonian(matrix_J, "J", self._basis_info)
        else:
            raise ValueError(f"Unknown units: {self._units}")
    
    def to_frequency_units(self) -> Hamiltonian:
        """周波数単位（rad/fs）に変換"""
        if self._units == "rad/fs":
            return Hamiltonian(self._matrix, "rad/fs", self._basis_info)
        elif self._units == "J":
            # J → rad/fs: ω = E/ℏ, J → rad/s → rad/fs
            matrix_rad_fs = self._matrix / self._HBAR * 1e-15
            return Hamiltonian(matrix_rad_fs, "rad/fs", self._basis_info)
        else:
            raise ValueError(f"Unknown units: {self._units}")
    
    def get_matrix(self, units: Literal["J", "rad/fs"] | None = None) -> np.ndarray:
        """
        指定した単位でハミルトニアン行列を取得
        
        Parameters
        ----------
        units : {"J", "rad/fs"}, optional
            取得したい単位。Noneの場合は現在の単位
            
        Returns
        -------
        np.ndarray
            ハミルトニアン行列
        """
        if units is None or units == self._units:
            return self.matrix
        elif units == "J":
            return self.to_energy_units().matrix
        elif units == "rad/fs":
            return self.to_frequency_units().matrix
        else:
            raise ValueError(f"Unknown units: {units}")
    
    def get_eigenvalues(self, units: Literal["J", "rad/fs"] | None = None) -> np.ndarray:
        """
        指定した単位で固有値を取得
        
        Parameters
        ----------
        units : {"J", "rad/fs"}, optional
            取得したい単位。Noneの場合は現在の単位
            
        Returns
        -------
        np.ndarray
            固有値
        """
        if units is None or units == self._units:
            return self.eigenvalues
        elif units == "J":
            return self.to_energy_units().eigenvalues
        elif units == "rad/fs":
            return self.to_frequency_units().eigenvalues
        else:
            raise ValueError(f"Unknown units: {units}")
    
    def energy_differences(self, units: Literal["J", "rad/fs"] | None = None) -> np.ndarray:
        """
        すべての固有値間のエネルギー差を計算
        
        Parameters
        ----------
        units : {"J", "rad/fs"}, optional
            取得したい単位。Noneの場合は現在の単位
            
        Returns
        -------
        np.ndarray
            エネルギー差の配列（重複なし、正の値のみ）
        """
        eigvals = self.get_eigenvalues(units)
        diffs = []
        for i in range(len(eigvals)):
            for j in range(i + 1, len(eigvals)):
                diff = abs(eigvals[i] - eigvals[j])
                if diff > 0:  # ゼロでない差のみ
                    diffs.append(diff)
        return np.array(diffs)
    
    def max_energy_difference(self, units: Literal["J", "rad/fs"] | None = None) -> float:
        """
        最大エネルギー差を取得
        
        Parameters
        ----------
        units : {"J", "rad/fs"}, optional
            取得したい単位。Noneの場合は現在の単位
            
        Returns
        -------
        float
            最大エネルギー差
        """
        diffs = self.energy_differences(units)
        return np.max(diffs) if len(diffs) > 0 else 0.0
    
    def __repr__(self) -> str:
        """文字列表現"""
        info = f"Hamiltonian({self.shape[0]}×{self.shape[1]}, units='{self.units}')"
        if self.is_diagonal():
            eigvals = self.eigenvalues
            if len(eigvals) <= 4:
                eigvals_str = ", ".join(f"{val:.3e}" for val in eigvals)
            else:
                eigvals_str = f"{eigvals[0]:.3e}, ..., {eigvals[-1]:.3e}"
            info += f"\n  Eigenvalues: [{eigvals_str}]"
        return info
    
    def __add__(self, other) -> Hamiltonian:
        """ハミルトニアンの加算"""
        if isinstance(other, Hamiltonian):
            if other.units != self.units:
                other = other.to_energy_units() if self.units == "J" else other.to_frequency_units()
            return Hamiltonian(self._matrix + other._matrix, self.units, self._basis_info)
        elif isinstance(other, (int, float, np.ndarray)):
            return Hamiltonian(self._matrix + other, self.units, self._basis_info)
        else:
            return NotImplemented
    
    def __sub__(self, other) -> Hamiltonian:
        """ハミルトニアンの減算"""
        if isinstance(other, Hamiltonian):
            if other.units != self.units:
                other = other.to_energy_units() if self.units == "J" else other.to_frequency_units()
            return Hamiltonian(self._matrix - other._matrix, self.units, self._basis_info)
        elif isinstance(other, (int, float, np.ndarray)):
            return Hamiltonian(self._matrix - other, self.units, self._basis_info)
        else:
            return NotImplemented
    
    def __mul__(self, scalar) -> Hamiltonian:
        """スカラー倍"""
        if isinstance(scalar, (int, float)):
            return Hamiltonian(self._matrix * scalar, self.units, self._basis_info)
        else:
            return NotImplemented
    
    def __rmul__(self, scalar) -> Hamiltonian:
        """スカラー倍（右から）"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar) -> Hamiltonian:
        """スカラー除算"""
        if isinstance(scalar, (int, float)):
            return Hamiltonian(self._matrix / scalar, self.units, self._basis_info)
        else:
            return NotImplemented 