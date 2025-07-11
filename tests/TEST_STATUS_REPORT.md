# テストスイートの現状レポート

## 1. 概要

- **総テスト数**: 313
- **合格 (Passed)**: 288
- **スキップ (Skipped)**: 10
- **期待された失敗 (XFAIL)**: 15
- **失敗 (Failed)**: 0

APIの不整合に関する主要な問題は解決され、テストスイートは安定した状態にあります。

## 2. テストカバレッジ

- **全体カバレッジ: 44%**

### モジュール別カバレッジ詳細（抜粋）

#### 🟢 高カバレッジ (80%以上)
| モジュール | カバレッジ |
| --- | --- |
| `core/basis/linmol.py` | 92% |
| `core/basis/states.py` | 94% |
| `core/basis/viblad.py` | 87% |
| `dipole/linmol/cache.py`| 86% |
| `dipole/vib/harmonic.py`| 100% |
| `dipole/vib/morse.py` | 100% |

#### 🟡 中程度カバレッジ (50-79%)
| モジュール | カバレッジ |
| --- | --- |
| `core/basis/twolevel.py` | 72% |
| `core/units/converters.py` | 63% |
| `core/units/parameter_processor.py`| 63% |
| `core/units/validators.py` | 61% |
| `core/propagation/utils.py`| 69% |

#### 🔴 低カバレッジ (<50%)
| モジュール | カバレッジ |
| --- | --- |
| `core/nondimensional/*` | 6-28% |
| `core/propagation/algorithms/*`| 13-28% |
| `core/propagation/liouville.py`| 20% |
| `core/propagator.py` | 18% |
| `simulation/runner.py`| 55% |
| `spectroscopy/*` | 12-71% |

## 3. 期待された失敗 (XFAIL) の一覧

これらのテストは、より根本的な数値計算上の問題や、複雑なロジックの不整合を示唆しており、今後の修正対象となります。

### `tests/test_integration.py`
- `test_full_simulation_workflow`: `Norm is not conserved, returns large number`
- `test_multi_level_excitation`: `Returns NaN`
- `test_mixed_vs_pure_states`: `AssertionError on rho comparison`
- `test_coherent_vs_incoherent`: `Returns NaN`
- `test_numerical_precision`: `Norm is not conserved`
- `test_field_strength_scaling`: `IndexError: invalid index to scalar variable.`

### `tests/test_nondimensional_consistency.py`
- `test_final_state_consistency`: `Nondimensionalization calculation is incorrect`
- `test_trajectory_consistency`: `Nondimensionalization calculation is incorrect`
- `test_weak_field_consistency`: `Nondimensionalization calculation is incorrect`

### `tests/test_performance.py`
- `test_very_large_system`: `Shape mismatch in return value`
- `test_long_time_evolution`: `Norm is not conserved in long-time evolution`
- `test_numerical_stability_large_system`: `Energy is not conserved in large system`

### `tests/test_propagator.py`
- `test_liouville_propagation`: `liouville_propagation returns NaN`
- `test_schrodinger_propagation_with_constant_polarization`: `Shape mismatch in return value`

### `tests/test_rk4_comprehensive.py`
- `test_minimal_system_size`: `Shape mismatch in return value`

## 4. 今後の課題

1.  **カバレッジの向上**: 特に`nondimensional`と`spectroscopy`モジュールのテストを拡充する。
2.  **XFAILテストの修正**:
    -   無次元化 (`nondimensional`) の物理的・数学的ロジックを再レビューする。
    -   長時間・大規模系での数値的安定性（ノルム・エネルギー保存）を改善する。
    -   `NaN`が発生する伝播の問題をデバッグする。

以上のレポートを`tests/TEST_STATUS_REPORT.md`として保存します。 