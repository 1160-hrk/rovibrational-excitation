from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Grid1D:
    x: np.ndarray  # shape (Nx,)

    def kx(self) -> np.ndarray:
        Nx = self.x.size
        dx = float(self.x[1] - self.x[0])
        return 2 * np.pi * np.fft.fftfreq(Nx, d=dx)


@dataclass
class Grid2D:
    x: np.ndarray  # shape (Nx,)
    y: np.ndarray  # shape (Ny,)

    def k(self) -> Tuple[np.ndarray, np.ndarray]:
        Nx = self.x.size
        Ny = self.y.size
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        return np.meshgrid(kx, ky, indexing="ij")


@dataclass
class Grid3D:
    x: np.ndarray  # shape (Nx,)
    y: np.ndarray  # shape (Ny,)
    z: np.ndarray  # shape (Nz,)

    def k(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Nx = self.x.size
        Ny = self.y.size
        Nz = self.z.size
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        dz = float(self.z[1] - self.z[0])
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
        return np.meshgrid(kx, ky, kz, indexing="ij")

