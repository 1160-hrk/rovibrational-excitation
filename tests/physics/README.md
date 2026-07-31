# Physics reference tests

This directory is reserved for trusted scientific references and invariants.
Every collected test here must use `@pytest.mark.physics` and state:

- the equation, invariant, or independently derived reference;
- all physical parameters and units;
- the expected tolerance and why it is appropriate;
- the contract or decision ID when one exists.

The files are introduced with real checks in the following Phase 0 tasks:

| File | Owner | Status | Required scope |
|---|---|---|---|
| `test_two_level_reference.py` | P0.3 | Implemented; CUDA parity pending | Analytic phase, density equivalence, constant drive, scalar polarization, dimensional/backend agreement |
| `test_vib_ladder_reference.py` | P0.4 | Planned | Energies, Morse derivation/bounds/isolation, scalar polarization |
| `test_linear_molecule_reference.py` | P0.5 | Planned | Indexing, energies, Cartesian polarization, dense/sparse behavior |
| `test_dipole_selection_rules.py` | P0.4/P0.5 | Planned | Harmonic, Morse, and rotational selection rules |
| `test_solver_invariants.py` | P0.6 | Planned | Convergence, norm/trace/Hermiticity, time sampling, capability errors |
| `test_dimensional_equivalence.py` | P0.3/P0.4 | TwoLevel covered in its model file | Cross-model dimensional and nondimensional result/time agreement |

Empty placeholder test modules are intentionally forbidden: a planned filename
appears only when at least one meaningful reference test is implemented.
Existing behavior-regression tests remain under `tests/` or
`tests/contracts/` until their scientific reference is independently
established.
