"""Freeze model/coupling projection before moving its contract owner."""

from types import SimpleNamespace

from rovibrational_excitation.dynamics.problem import Axis, CouplingSpec, SystemModel
from rovibrational_excitation.models.factory import ModelComponents


def test_model_components_projection_preserves_objects_and_coupling_order():
    basis = SimpleNamespace(size=lambda: 2)
    state = object()
    hamiltonian = SimpleNamespace(size=2)
    dipole = SimpleNamespace(basis=basis)
    coupling = CouplingSpec.cartesian("zx")
    metadata = {"source": "caller"}
    components = ModelComponents(
        name="characterization",
        basis=basis,
        state=state,
        hamiltonian=hamiltonian,
        dipole=dipole,
        coupling=coupling,
        metadata=metadata,
    )

    model = components.to_system_model()

    assert isinstance(model, SystemModel)
    assert model.name == components.name
    assert model.basis is basis
    assert model.hamiltonian is hamiltonian
    assert model.dipole is dipole
    assert model.coupling is coupling
    assert model.dimension == 2
    assert model.coupling.axes == ("z", "x")
    assert model.coupling.propagation_kwargs() == {"axes": "zx"}
    metadata["source"] = "changed"
    assert dict(model.metadata) == {"source": "caller"}


def test_scalar_coupling_projection_keeps_existing_axis_labels():
    assert CouplingSpec.scalar(Axis.X).propagation_kwargs() == {"coupling_axis": "x"}
    assert CouplingSpec.scalar(Axis.Z).propagation_kwargs() == {"coupling_axis": "z"}
