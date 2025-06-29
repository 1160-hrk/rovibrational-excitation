# rovibrational-excitation ドキュメント

このディレクトリには、rovibrational-excitation パッケージの詳細なドキュメントが含まれています。

## 📖 ドキュメント一覧

### 🔧 設定・パラメータ

| ドキュメント | 内容 | 対象者 |
|-------------|------|--------|
| **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** | **全パラメータの詳細リファレンス** | 全ユーザー |
| [SWEEP_SPECIFICATION.md](SWEEP_SPECIFICATION.md) | パラメータスイープ仕様 | 中級ユーザー |

### 📊 使用例・チュートリアル

| リソース | 内容 | レベル |
|----------|------|-------|
| [../examples/](../examples/) | パラメータファイル例 | 初級～中級 |
| [../tests/](../tests/) | テストコード例 | 上級 |

## 🚀 クイックスタート

### 1. 基本的な使い方

**まずはパラメータリファレンスから始めましょう**：

```bash
# 1. パラメータファイルを作成
cp examples/params_template.py my_params.py

# 2. パラメータを編集（PARAMETER_REFERENCE.mdを参照）
vim my_params.py

# 3. シミュレーション実行
python -m rovibrational_excitation.simulation.runner my_params.py
```

### 2. 設定手順

1. **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** で必須パラメータを確認
2. 物理系に応じてパラメータを設定
3. 必要に応じてスイープ設定を追加
4. テスト実行でエラーがないことを確認

## 📋 カテゴリ別ガイド

### 初心者向け
1. **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** の「基本例」から開始
2. `examples/` フォルダの簡単な例を試す
3. 小さな系（`V_max=2, J_max=2`）で動作確認

### 中級者向け
1. **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** の「スイープ例」を参照
2. [SWEEP_SPECIFICATION.md](SWEEP_SPECIFICATION.md) でスイープ制御を学習
3. パフォーマンス最適化を実践

### 上級者向け
1. **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** の「高度な設定例」を活用
2. `backend="cupy"` でGPU計算を試す
3. カスタム包絡線関数やスパース行列を使用

## 🔍 目的別ガイド

### CO2分子の励起シミュレーション
```python
# PARAMETER_REFERENCE.mdの基本例をベースに
omega_rad_phz = 2349 * 2 * np.pi * 3e10 / 1e15  # ν3 mode
mu0_Cm = 0.3 * 3.33564e-30                      # ~0.3 Debye
V_max, J_max = 3, 5                              # 適度なサイズ
```

### パラメータスイープ
```python
# 複数条件の比較
duration = [10.0, 20.0, 30.0]           # パルス幅
amplitude_sweep = [1e8, 5e8, 1e9]       # 電場強度（明示的スイープ）
polarization = [1.0, 0.0]               # 固定値
```

### 高強度レーザー計算
```python
# 強電場・非線形効果
amplitude = 1e12                         # 極強電場
V_max, J_max = 10, 20                    # 大きな基底
backend = "cupy"                         # GPU加速
```

### 超短パルス計算
```python
# フェムト秒パルス
duration = 5.0                           # 5fs FWHM
dt = 0.01                               # 細かい時間刻み
gdd = 100.0                             # 群遅延分散
```

## 💡 よく使われる設定パターン

### 高速テスト用
```python
# 開発・デバッグ用の軽量設定
V_max, J_max = 1, 1
t_start, t_end, dt = -10.0, 10.0, 0.2
sample_stride = 5
save = False
```

### 本格計算用
```python
# 発表・論文用の高精度設定
V_max, J_max = 5, 10
t_start, t_end, dt = -100.0, 100.0, 0.05
sample_stride = 1
backend = "cupy"  # GPU使用
```

### バッチ処理用
```python
# 大量ケースの並列処理
checkpoint_interval = 5    # 頻繁なチェックポイント
nproc = 8                  # 並列数
dry_run = True             # まずケース数確認
```

## 🛠️ トラブルシューティング

### よくある問題と解決法

1. **パラメータエラー**
   - **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** の必須パラメータをチェック
   - `examples/` の動作例と比較

2. **メモリ不足**
   - `V_max`, `J_max` を小さくする
   - `sample_stride` を増やす
   - `dense=False` でスパース行列を使用

3. **計算が遅い**
   - **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** の「パフォーマンス最適化」を参照
   - GPU環境なら `backend="cupy"`
   - 並列実行 `nproc=8`

4. **スイープエラー**
   - [SWEEP_SPECIFICATION.md](SWEEP_SPECIFICATION.md) でスイープルールを確認
   - `--dry-run` でケース数をチェック

## 📂 ディレクトリ構造

```
docs/
├── README.md                    # このファイル（ドキュメント概要）
├── PARAMETER_REFERENCE.md       # 全パラメータリファレンス ⭐
└── SWEEP_SPECIFICATION.md       # スイープ仕様詳細

examples/
├── params_template.py           # 基本テンプレート
├── params_CO2_AntiSymm.py      # CO2励起例  
├── params_example_new_sweep.py # 新スイープ仕様例
└── params_example_checkpoint.py # チェックポイント例

tests/
├── test_*.py                   # テストコード（参考例）
└── README.md                   # テスト実行方法
```

## 🔗 関連リンク

- **メインパッケージ**: [../src/rovibrational_excitation/](../src/rovibrational_excitation/)
- **使用例**: [../examples/](../examples/)
- **テスト**: [../tests/](../tests/)
- **GitHub**: プロジェクトリポジトリ（URL設定により）

## 📝 更新履歴

- **v1.3** (2024): パラメータリファレンス追加、スイープ仕様改善
- **v1.2** (2024): チェックポイント機能追加
- **v1.1** (2024): 新スイープ仕様導入
- **v1.0** (2024): 初期リリース

---

**💡 ヒント**: まずは **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** を読んで、パラメータの全体像を把握することをお勧めします。 