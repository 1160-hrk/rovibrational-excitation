# Physics Validation Scripts / 物理検証スクリプト

このディレクトリには、`rovibrational_excitation`パッケージの各モジュールが物理的に正しく動作することを確認するための検証スクリプトが含まれています。

## 📁 ディレクトリ構造

```
validation/
├── README.md                               # このファイル
├── check_memory_eigendecomposition.py     # メモリ・パフォーマンス検証
├── core/                                   # コア機能の物理検証
│   ├── check_core_basis.py                # 基底クラスの物理的妥当性
│   ├── check_core_states.py               # 状態ベクトル・密度行列
│   ├── check_core_hamiltonian.py          # ハミルトニアン生成
│   ├── check_core_electric_field.py       # 電場生成・変調
│   ├── check_core_propagator.py           # 高レベル伝播関数
│   ├── check_core__rk4_schrodinger.py     # RK4シュレディンガー伝播
│   ├── check_core__rk4_schrodinger_old.py # RK4レガシー実装比較
│   └── check_core__splitop_schrodinger.py # Split-Operator伝播
├── dipole/                                 # 双極子行列の物理検証
│   ├── check_dipole_builder.py            # 双極子行列構築
│   ├── check_sparse_dipole.py             # スパース行列実装
│   └── check_cache.py                     # キャッシュ機能
└── simulation/                             # シミュレーション統合検証
    └── check_simulation_runner.py         # バッチ実行システム
```

## 🎯 各検証スクリプトの目的

### Core Physics (コア物理)

- **基底と状態**: 量子状態の直交性、規格化、物理的選択則の確認
- **ハミルトニアン**: エネルギー固有値、エルミート性、スペクトル特性
- **電場**: 包絡線関数、周波数変調、分散補正の物理的妥当性
- **伝播アルゴリズム**: ユニタリ性、エネルギー保存、数値収束性

### Dipole Matrices (双極子行列)

- **選択則**: 振動・回転遷移の選択則（Δv=±1, ΔJ=±1）
- **対称性**: エルミート性、パリティ選択則
- **スケーリング**: 双極子モーメントの物理的スケール

### Simulation Integration (統合シミュレーション)

- **バッチ処理**: パラメータスイープ、並列実行、結果整合性
- **パフォーマンス**: メモリ使用量、計算時間、数値精度

## 🚀 実行方法

### 個別検証

```bash
# コア機能の検証
python validation/core/check_core_basis.py
python validation/core/check_core_propagator.py

# 双極子行列の検証
python validation/dipole/check_dipole_builder.py

# シミュレーション統合の検証
python validation/simulation/check_simulation_runner.py
```

### 全体検証

```bash
# すべての検証を実行
find validation/ -name "check_*.py" -exec python {} \;

# または、特定カテゴリのみ
find validation/core/ -name "check_*.py" -exec python {} \;
```

## 📊 期待される出力

各検証スクリプトは以下のような情報を出力します：

```
✅ 物理的妥当性チェック: PASSED
📊 数値精度: 1.2e-12 (tolerance: 1e-10)
⚡ 計算時間: 0.45秒
💾 メモリ使用量: 12.3 MB
🔬 物理量保存: エネルギー ±0.01%, ノルム ±1e-14
```

## 🧪 テストとの違い

| 項目 | Unit Tests (`tests/`) | Physics Validation (`validation/`) |
|------|----------------------|-----------------------------------|
| **目的** | コード機能の正確性 | 物理法則の遵守 |
| **チェック対象** | 関数の入出力 | 保存則、選択則、対称性 |
| **実行速度** | 高速（秒単位） | 中程度（分単位も可） |
| **データサイズ** | 小規模（テスト用） | 現実的サイズ |
| **CI/CD** | 必須 | 推奨（重要な変更時） |

## 🔧 開発者向け

### 新しい検証スクリプトの追加

1. **命名規則**: `check_[module_name].py`
2. **出力形式**: 統一されたメッセージフォーマット
3. **物理チェック**: 関連する保存則・選択則を必ず確認
4. **エラーハンドリング**: 物理的に不正な結果の場合は明確にエラー表示

### 検証基準

```python
# 例: エネルギー保存の確認
energy_initial = np.real(psi0.conj().T @ H0 @ psi0)
energy_final = np.real(psi_final.conj().T @ H0 @ psi_final)
energy_conservation = abs(energy_final - energy_initial) / abs(energy_initial)

assert energy_conservation < 1e-10, f"Energy not conserved: {energy_conservation:.2e}"
print(f"✅ Energy conservation: {energy_conservation:.2e}")
```

## 📈 継続的検証

開発プロセスでの検証タイミング：

1. **新機能追加時**: 関連する検証スクリプトを実行
2. **リリース前**: 全検証スクリプトの実行
3. **定期チェック**: 月次での全体検証
4. **パフォーマンス検証**: 大きな変更後のベンチマーク

物理検証は科学計算ソフトウェアの信頼性の根幹です。定期的な実行を推奨します。 