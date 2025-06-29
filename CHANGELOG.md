# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Added
- Version consistency checks between pyproject.toml and Git tags
- Automated release workflow via GitHub Actions
- Release management script (`scripts/release.py`)

### 🔧 Changed
- Improved test coverage documentation

### 🐛 Fixed
- Minor bug fixes in propagation algorithms

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

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Run release script: `python scripts/release.py X.Y.Z "Release message"`
4. GitHub Actions automatically builds and publishes to PyPI

### Breaking Changes

Major version increments indicate breaking changes. Always review the changelog
and migration guide when upgrading across major versions.

---

## Links

- [PyPI Releases](https://pypi.org/project/rovibrational-excitation/#history)
- [GitHub Releases](https://github.com/1160-hrk/rovibrational-excitation/releases)
- [GitHub Tags](https://github.com/1160-hrk/rovibrational-excitation/tags) 