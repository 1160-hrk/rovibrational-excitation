"""Explicit propagation-option builders used only by tests."""

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.core.propagation.capabilities import (
    PropagationAlgorithm,
)
from rovibrational_excitation.core.propagation.options import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)


def propagation_options(
    *,
    algorithm: PropagationAlgorithm = PropagationAlgorithm.RK4,
    backend: ArrayBackend = ArrayBackend.NUMPY,
    storage: MatrixStorage = MatrixStorage.DENSE,
    return_trajectory: bool,
    sample_stride: int = 1,
    scaling: ScalingMode = ScalingMode.DIMENSIONAL,
    renormalization: RenormalizationPolicy = RenormalizationPolicy.DISABLED,
) -> PropagationOptions:
    """Build a fully explicit production options object for one test case."""
    return PropagationOptions(
        algorithm=algorithm,
        execution=ExecutionPolicy(backend=backend, storage=storage),
        return_trajectory=return_trajectory,
        sample_stride=sample_stride,
        scaling=scaling,
        renormalization=renormalization,
    )


__all__ = ["propagation_options"]
