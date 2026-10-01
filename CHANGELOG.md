# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Added
- Final-version consistency checks between pyproject.toml and Git tags
- Release workflow with complete CPU gates and mandatory real-CUDA evidence
- Manual pre-tag CUDA workflow and schema-v1 parity/backend/transfer/timing
  evidence recorder; accepted JSON is retained and attached to final releases
- Required normal/release development-container smoke with a minimal build
  context, non-root package import, and authenticated Jupyter API check
- Normal, CUDA, and release workflows pin every external Action to a reviewed
  immutable release commit
- PyPI publication uses isolated, job-scoped OIDC Trusted Publishing with no
  long-lived API-token input or fallback
- Explicit dry-run/apply release preparation script (`scripts/release.py`)
  whose successful apply is a non-acceptance handoff and never suggests tagging
  before the external release gates pass
- CI smoke coverage for the supported parameter template and example index
- Reproducible actionlint/ShellCheck workflow lint plus GitHub Check
  annotations for failed JUnit cases and container-smoke stages

### 🔧 Changed
- The package root now exposes only the exact lazy typed v0.3 API; v0.2 root
  convenience names have no compatibility shims.
- English and Japanese READMEs now use executable unit-explicit examples and
  state the verified CPU, unverified CUDA, SymTop, optimizer, spectroscopy, and
  persistence boundaries.
- The documentation index now distinguishes verified current contracts from
  guides still undergoing v0.3 migration audit.
- A v0.2-to-v0.3 migration guide now covers moved imports, required units,
  field/time-grid contracts, distinct optimizer layouts, spectroscopy, and
  strict result/checkpoint handling without inferred conversions.
- The normal-simulation parameter reference now follows the strict generated,
  sampled-field, model, unit, and execution schemas.
- The sweep guide now documents exact singleton, insertion-order,
  Cartesian-product, result-path, dry-run, and resume-provenance behavior.
- The time-propagation guide now documents only implemented RK4/split paths,
  exact timing/output semantics, capability failures, and CUDA limitations.
- The unit guide now fixes caller provenance, canonical conversion, supported
  spellings, frequency 2π, field/intensity, optimizer, and spectroscopy rules.
- The development image no longer writes tokenless, wildcard-bind Jupyter
  configuration and installs project dependencies from `pyproject.toml`.
- Removed unwired Codecov configuration and setup claims; branch coverage is
  enforced locally in CI and retained as report/XML artifacts.
- Repository-wide Markdown links, code fences, and YAML syntax are checked by
  contracts; the required CI quality job runs pinned, checksum-verified
  actionlint.
- `pyproject.toml` is now the only dependency manifest; stale duplicate
  requirements files and unrelated spectroscopy version/contact metadata were
  removed, and the test guide matches the actual CI policy.
- A Phase 8 readiness audit records passing local CPU, coverage, quality,
  documentation, build/Twine, isolated-wheel, CLI, and payload checks while
  keeping real CUDA, container, publication, and final-version gates explicit.
- CuPy RK4 now uses the CPU-consistent `H0 - mu E` four-stage graph, and
  CuPy split preserves its static Cartesian, rotating Cartesian, and
  helicity-projected formulae in a separate device-native owner. Both honor
  trajectory/stride/renormalization and require real-GPU evidence.
- Release preparation never commits, tags, pushes, publishes, or prompts
  implicitly; publication remains an explicit GitHub release workflow action.
- Jupyter binds to localhost with standard authentication by default and no
  longer writes global user configuration.
- The generated example index scans only supported top-level examples and
  cannot include the historical archive.
- Improved test coverage documentation
- Development-branch normal results now publish complete manifest-v1 bundles
  through an atomic generation pointer; valid direct-layout v1 results remain
  readable but require explicit migration before overwrite.
- Checkpoint and failure-list JSON are published as one immutable generation
  through an atomic pointer.
- Checkpoint payload schema v1 strictly validates the selected pair and binds
  resume to the complete ordered expanded-run declaration with SHA-256.
  Corrupt, unversioned, unknown, or different-run checkpoints now raise before
  case execution instead of falling back, printing-and-returning `None`, or
  upgrading implicitly.
- Standalone result-directory plots now load `t_E/E` and `t_p/pop` only through
  the strict published-result schema-v1 reader. Legacy NPY collections,
  malformed publications, and scalar input to the Cartesian-vector plot raise
  explicitly instead of falling back or printing and returning.
- P7.2 persistence acceptance now fixes the result/checkpoint schema authorities
  and records the exact durability, provenance, migration, and release limits
  that remain outside this checkpoint.

### 🐛 Fixed
- Minor bug fixes in propagation algorithms

## [0.3.0.dev1] - 2026-09-21

Development checkpoint for the `refactor/v0.3` branch; not a final release or
Git tag. P7.1 simulation-runner decomposition has passed its acceptance audit
without changing established physical calculations. Typed propagation, units,
and consolidated CPU models are in place. The result schema, optimization and
spectroscopy decomposition, public root API, and device-native CUDA with a
real-GPU run remain required before `0.3.0`.

## [0.1.4] - 2024-12-XX

### 🚀 Added
- Comprehensive test suite with 75% coverage
- LinMolBasis, TwoLevelBasis, VibLadderBasis classes
- Electric field generation with advanced features
- GPU acceleration support via CuPy
- Batch simulation runner

### 🔧 Changed
- Refactored basis classes to use abstract base class
- Improved error handling in propagation functions

### 🐛 Fixed
- Memory optimization in RK4 propagators
- Numerical precision improvements

### 📚 Documentation
- Complete parameter reference documentation
- Test coverage reports and guides
- Development setup instructions

## [0.1.3] - 2024-XX-XX

### 🚀 Added
- Initial split-operator propagator
- Dipole matrix caching system

### 🔧 Changed
- Performance optimizations in core algorithms

### 🐛 Fixed
- Bug fixes in basis generation

## [0.1.2] - 2024-XX-XX

### 🚀 Added
- RK4 Liouville-von Neumann propagator
- Enhanced electric field modulation

### 🔧 Changed
- Code style improvements with Black and Ruff

## [0.1.1] - 2024-XX-XX

### 🚀 Added
- Basic RK4 Schrödinger propagator
- Linear molecule basis class

### 🐛 Fixed
- Initial bug fixes and stability improvements

## [0.1.0] - 2024-XX-XX

### 🚀 Added
- Initial release
- Core package structure
- Basic quantum dynamics functionality

---

## Version Guidelines

### Semantic Versioning

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New functionality in backward-compatible manner  
- **PATCH** (0.0.X): Backward-compatible bug fixes

### Release Types

- 🚀 **Added**: New features
- 🔧 **Changed**: Changes in existing functionality
- 📚 **Documentation**: Documentation improvements
- 🐛 **Fixed**: Bug fixes
- 🗑️ **Deprecated**: Soon-to-be removed features
- ❌ **Removed**: Removed features
- 🔒 **Security**: Security improvements

### Release Process

1. Complete the Phase 8 acceptance gates and update this CHANGELOG.md.
2. Validate the transition with `python scripts/release.py X.Y.Z --dry-run`.
3. From a clean worktree, run `python scripts/release.py X.Y.Z --apply`.
4. Review and commit the version change explicitly.
5. Confirm the real-GPU runner, protected PyPI environment, and exact Trusted
   Publisher identity are available.
6. Create and push the annotated `vX.Y.Z` tag explicitly; GitHub Actions
   publishes only after every CPU, CUDA, build, and clean-wheel gate passes.

### Breaking Changes

Major version increments indicate breaking changes. Always review the changelog
and migration guide when upgrading across major versions.

---

## Links

- [PyPI Releases](https://pypi.org/project/rovibrational-excitation/#history)
- [GitHub Releases](https://github.com/1160-hrk/rovibrational-excitation/releases)
- [GitHub Tags](https://github.com/1160-hrk/rovibrational-excitation/tags) 