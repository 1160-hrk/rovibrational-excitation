"""Explicit array-backend and matrix-storage execution choices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArrayBackend(str, Enum):
    """Array ownership selected for one calculation."""

    NUMPY = "numpy"
    CUPY = "cupy"


class MatrixStorage(str, Enum):
    """Operator storage selected for one calculation."""

    DENSE = "dense"
    CSR = "csr"


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Required backend/storage pair shared by construction and propagation.

    The constructor accepts enum members only. Configuration adapters must use
    :meth:`from_strings`, so unknown spellings fail at one explicit boundary.
    """

    backend: ArrayBackend
    storage: MatrixStorage

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ArrayBackend):
            raise TypeError("backend must be an ArrayBackend")
        if not isinstance(self.storage, MatrixStorage):
            raise TypeError("storage must be a MatrixStorage")

    @classmethod
    def from_strings(cls, *, backend: str, storage: str) -> ExecutionPolicy:
        """Parse required configuration strings without fallback or inference."""
        try:
            backend_kind = ArrayBackend(backend)
        except ValueError:
            raise ValueError("backend must be 'numpy' or 'cupy'") from None
        try:
            storage_kind = MatrixStorage(storage)
        except ValueError:
            raise ValueError("storage must be 'dense' or 'csr'") from None
        return cls(backend=backend_kind, storage=storage_kind)

    @property
    def sparse(self) -> bool:
        """Return the legacy solver adapter flag."""
        return self.storage is MatrixStorage.CSR

    @property
    def dense(self) -> bool:
        """Return the legacy model-builder adapter flag."""
        return self.storage is MatrixStorage.DENSE


__all__ = ["ArrayBackend", "ExecutionPolicy", "MatrixStorage"]
