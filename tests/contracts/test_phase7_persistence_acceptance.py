"""Acceptance wiring for the Phase 7.2 persistence authorities."""

from rovibrational_excitation.io import checkpoint, result_schema, storage
from rovibrational_excitation.simulation import (
    batch,
    result_persistence,
    resume,
    runner,
)
from rovibrational_excitation.visualization import result_data


def test_phase7_persistence_paths_use_one_result_and_checkpoint_authority():
    assert storage.load_simulation_result is result_schema.load_simulation_result
    assert result_data.load_simulation_result is result_schema.load_simulation_result

    assert (
        result_persistence.create_result_generation
        is result_schema.create_result_generation
    )
    assert (
        result_persistence.write_result_manifest is result_schema.write_result_manifest
    )
    assert (
        result_persistence.publish_result_generation
        is result_schema.publish_result_generation
    )

    assert runner.CheckpointManager is checkpoint.CheckpointManager
    assert batch.CheckpointManager is checkpoint.CheckpointManager
    assert resume.CheckpointManager is checkpoint.CheckpointManager
