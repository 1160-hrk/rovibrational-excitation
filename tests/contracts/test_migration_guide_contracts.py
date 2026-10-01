"""Contracts for the intentionally breaking v0.3 migration guide."""

from __future__ import annotations

from pathlib import Path

import rovibrational_excitation as rve
from rovibrational_excitation.core.operators import Hamiltonian
from rovibrational_excitation.core.states import (
    DensityState,
    IncoherentEnsemble,
    PureState,
)
from rovibrational_excitation.models.linear_molecule import (
    LinMolBasis,
    LinMolDipoleMatrix,
)
from rovibrational_excitation.spectroscopy import (
    AbsorbanceCalculator,
    ExperimentalConditions,
    create_calculator_from_params,
)

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "MIGRATION_V0_3.md"


def test_migration_guide_is_linked_from_each_public_document_entry():
    assert GUIDE.is_file()
    assert "docs/MIGRATION_V0_3.md" in (ROOT / "README.md").read_text()
    assert "docs/MIGRATION_V0_3.md" in (ROOT / "README_JP.md").read_text()
    assert "MIGRATION_V0_3.md" in (ROOT / "docs" / "README.md").read_text()


def test_documented_import_targets_are_real_and_removed_root_names_stay_removed():
    assert LinMolBasis.__module__.startswith(
        "rovibrational_excitation.models.linear_molecule"
    )
    assert LinMolDipoleMatrix.__module__.startswith(
        "rovibrational_excitation.models.linear_molecule"
    )
    assert Hamiltonian.__module__ == "rovibrational_excitation.core.operators"
    assert PureState.__module__ == "rovibrational_excitation.core.states"
    assert DensityState.__module__ == "rovibrational_excitation.core.states"
    assert IncoherentEnsemble.__module__ == "rovibrational_excitation.core.states"
    assert AbsorbanceCalculator.__module__.startswith(
        "rovibrational_excitation.spectroscopy"
    )
    assert ExperimentalConditions.__module__.startswith(
        "rovibrational_excitation.spectroscopy"
    )
    assert create_calculator_from_params.__module__.startswith(
        "rovibrational_excitation.spectroscopy"
    )
    for removed in (
        "LinMolBasis",
        "LinMolDipoleMatrix",
        "Hamiltonian",
        "StateVector",
        "DensityMatrix",
        "AbsorbanceCalculator",
        "ExperimentalConditions",
        "create_calculator_from_params",
    ):
        assert removed not in rve.__all__


def test_migration_guide_records_exact_normal_schema_replacements():
    text = GUIDE.read_text()
    for required in (
        '`use_M=True` | LinMol `representation="m_resolved"`',
        '`use_M=False` | LinMol `representation="m_incoherent_average"`',
        "`mu0_Cm` | `dipole_scale` + `dipole_scale_units`",
        "`pulse_duration` | `duration` + `duration_units`",
        'boolean `dense` / `sparse` | required `storage="dense"` / `storage="csr"`',
        "`auto_timestep`, `target_accuracy` | 削除",
        "旧 `carrier_freq_sin_mod` の数値を delay として流用する",
        "一対一変換はありません",
        "`run_simulation_case(params, field=...)`",
        "`propagation_dt = 2 * field_dt`",
    ):
        assert required in text


def test_migration_guide_keeps_optimizer_time_layouts_distinct():
    text = GUIDE.read_text()
    assert "旧 Krotov と標準 Krotov は機械的に置換しない" in text
    assert "`algorithm: legacy_batch_overlap`" in text
    assert "`algorithm: krotov`" in text
    assert "`field_dt_fs`" in text
    assert "`control_dt_fs`" in text
    assert "`lambda_a_units`" in text
    assert "Local: frozen legacy grid" in text
    assert "端点、segment index、共有境界、RK4 prefix は変更しない" in text


def test_migration_guide_forbids_implicit_historical_data_upgrade():
    text = GUIDE.read_text()
    assert "unversioned checkpoint は resume しない" in text
    assert "旧 layout へ fallback しない" in text
    assert "暗黙上書きしない" in text
    assert "一般的な自動 migration tool はない" in text
    assert "v0.3 入力で再計算する" in text
    assert "examples/archives/v0_2/optimization_configs/" in text
