"""CPU-verifiable acceptance contracts for the numerical dynamics engine."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import rovibrational_excitation.dynamics.result as result_module
from rovibrational_excitation.core.execution import ArrayBackend
from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.dynamics.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)
from tests.propagation_options import propagation_options
from tests.propagation_problem import propagation_problem


class _ExistingDeviceState:
    def __init__(self, values: np.ndarray):
        self.values = values
        self.shape = values.shape

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "typestr": self.values.dtype.str,
            "data": (1, False),
            "version": 3,
        }


def test_numpy_split_operator_executes_csr_inputs_through_dense_spectral_path() -> None:
    h0 = np.diag([0.0, 0.7]).astype(np.complex128)
    mu_x = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.complex128)
    mu_y = np.array([[0.0, -0.2j], [0.2j, 0.0]], dtype=np.complex128)
    field_x = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    field_y = 0.5 * field_x
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    arguments = (h0, mu_x, mu_y, field_x, field_y, initial, 0.02)

    dense = splitop_schrodinger(
        *arguments,
        return_traj=True,
        sparse=False,
    )
    csr = splitop_schrodinger(
        sp.csr_matrix(h0),
        sp.csr_matrix(mu_x),
        sp.csr_matrix(mu_y),
        field_x,
        field_y,
        initial,
        0.02,
        return_traj=True,
        sparse=True,
    )

    assert isinstance(csr, np.ndarray)
    np.testing.assert_array_equal(csr, dense)


def test_result_boundary_keeps_an_existing_device_state_without_conversion(
    monkeypatch,
) -> None:
    problem = propagation_problem(PureState(np.array([1.0, 0.0], dtype=np.complex128)))
    options = propagation_options(
        backend=ArrayBackend.CUPY,
        return_trajectory=False,
    )
    state = _ExistingDeviceState(np.array([1.0, 0.0], dtype=np.complex128))

    def fail_import(_name: str):
        raise AssertionError("an existing device state must not import or convert")

    monkeypatch.setattr(result_module, "import_module", fail_import)
    result = result_module.finalize_propagation_result(
        problem=problem,
        options=options,
        times_fs=np.array([problem.time_grid.t_end_fs]),
        state=state,
        state_kind="wavefunction",
        scales=None,
    )

    assert result.backend == "cupy"
    assert result.state is state


def test_split_operator_imports_required_numba_without_hidden_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    source_path = (
        root
        / "src"
        / "rovibrational_excitation"
        / "dynamics"
        / "algorithms"
        / "split_operator"
        / "schrodinger.py"
    )
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))

    direct_numba_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "numba"
    ]
    assert len(direct_numba_imports) == 1
    assert "_HAS_NUMBA" not in source
    assert "Dummy decorator" not in source


def test_cupy_rk4_has_a_separate_device_native_owner() -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper_path = (
        root
        / "src"
        / "rovibrational_excitation"
        / "dynamics"
        / "algorithms"
        / "rk4"
        / "schrodinger.py"
    )
    kernel_path = wrapper_path.with_name("schrodinger_cupy.py")

    wrapper_source = wrapper_path.read_text()
    kernel_source = kernel_path.read_text()

    assert "_KERNEL_SRC_TEMPLATE" not in wrapper_source
    assert "from .schrodinger_cupy import rk4_schrodinger_cupy" in wrapper_source
    assert ".get(" not in kernel_source
    assert "cp.asnumpy" not in kernel_source
    assert "H0 - field_x * mu_x - field_y * mu_y" in kernel_source


def test_cupy_split_has_a_separate_device_native_owner() -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper_path = (
        root
        / "src"
        / "rovibrational_excitation"
        / "dynamics"
        / "algorithms"
        / "split_operator"
        / "schrodinger.py"
    )
    kernel_path = wrapper_path.with_name("schrodinger_cupy.py")

    wrapper_source = wrapper_path.read_text()
    kernel_source = kernel_path.read_text()

    assert "from .schrodinger_cupy import splitop_schrodinger_cupy" in wrapper_source
    assert "def _splitop_static_cupy" not in wrapper_source
    assert "def _splitop_rotating_xy_cupy" not in wrapper_source
    assert "cp.asnumpy" not in wrapper_source
    assert "cp.asnumpy" not in kernel_source
    assert ".get(" not in kernel_source
