# 振動励起システム2次元パラメータマッピング

このディレクトリには、振動励起システム（VibLadderBasis）における電場振幅とパルス時間幅の2次元パラメータスイープを実行するためのサンプルファイルが含まれています。

## ファイル構成

### メインファイル
- `example_vibrational_2d_map.py`: 振動励起システム用の2次元パラメータマッピング（本格版）
- `test_vibrational_2d_map.py`: 簡易テスト版（少ない計算点で動作確認）

### 機能

1. **電場振幅とパルス時間幅の2次元スイープ**
   - 電場振幅: 1e8 - 1e10 V/m（設定可能）
   - パルス時間幅: 30 - 100 fs（設定可能）
   - 並列計算対応（マルチプロセス）

2. **振動励起の物理**
   - VibLadderBasis（純粋振動系、回転なし）
   - VibLadderDipoleMatrix（主にz方向遷移）
   - 調和振動子またはモース振動子ポテンシャル

3. **可視化**
   - imshow版2次元マップ
   - pcolormesh版2次元マップ  
   - contour版等高線プロット
   - パルス面積条件線の重ね合わせ表示

## 使用方法

### 1. 簡易テスト版の実行
```bash
cd examples
python test_vibrational_2d_map.py
```

### 2. 本格版の実行
```bash
cd examples  
python example_vibrational_2d_map.py
```

### 3. パラメータの調整

`example_vibrational_2d_map.py`の上部にある設定セクションで調整可能：

```python
# 2次元スイープ範囲設定
DURATION_MIN = 30.0   # パルス時間幅の最小値 [fs]
DURATION_MAX = 100.0  # パルス時間幅の最大値 [fs]
DURATION_POINTS = 80  # パルス時間幅の点数

AMPLITUDE_MIN = 1e8   # 電場振幅の最小値 [V/m]
AMPLITUDE_MAX = 1e10  # 電場振幅の最大値 [V/m]
AMPLITUDE_POINTS = 80 # 電場振幅の点数

# システムパラメータ（SimulationConfigクラス内）
V_max: int = 3                          # 最大振動量子数
omega_01: float = 2349.1                # 振動周波数 [cm^-1]
delta_omega: float = 25.0               # 非調和性補正 [cm^-1]
potential_type: str = "harmonic"        # "harmonic" or "morse"
```

### 4. 並列計算設定

```python
# 並列計算設定
MAX_WORKERS = min(8, mp.cpu_count())    # CPUコア数
CHUNK_SIZE = 500                        # バッチサイズ
```

## 出力

### 保存先
- `examples/figures/` ディレクトリ
- ファイル名に実行日時が自動付与

### 生成されるファイル
- `vibrational_2d_imshow_YYYYMMDD_HHMMSS.png`
- `vibrational_2d_pcolormesh_YYYYMMDD_HHMMSS.png`
- `vibrational_2d_contour_YYYYMMDD_HHMMSS.png`

### プロット内容
- **カラーマップ**: 最終励起状態確率（v≥1の占有確率）
- **条件線**: パルス面積条件 `MU0*amplitude * duration * sqrt(2*pi)/HBAR = (2n+1)*pi`
- **軸**: x軸=パルス時間幅[fs], y軸=電場振幅[V/m]

## 物理的解釈

### パルス面積条件線
グラフ上の破線は以下の条件を満たすパラメータ組み合わせ：
```
パルス面積 = μ₀ × E₀ × τ × √(2π) / ℏ = (2n+1)π
```
ここで：
- μ₀: 双極子行列要素
- E₀: 電場振幅
- τ: パルス時間幅
- n: 整数（0, 1, 2, ...）

### 振動励起効率
- **強い励起**: 条件線付近（特にn=0, 1, 2...）
- **弱い励起**: 条件線から離れた領域
- **共鳴条件**: キャリア周波数が振動周波数と一致

## 技術的詳細

### 基底系
- `VibLadderBasis`: 純粋振動系（|v=0⟩, |v=1⟩, |v=2⟩, ...）
- 回転自由度なし、磁気量子数なし

### 双極子遷移
- 主にz方向成分（VibLadderDipoleMatrixの特性）
- 現在の実装ではx偏光 + zx軸の組み合わせで計算
- 選択則: Δv = ±1

### 数値計算
- 時間発展: schrodinger_propagation
- 並列化: ProcessPoolExecutor（バッチ処理）
- エラー処理: 個別ケース失敗時のフォールバック

## 注意事項

1. **計算時間**: フル解像度（80×80点）では数時間〜数十時間
2. **メモリ使用量**: 大きな基底（V_max > 5）では注意
3. **収束性**: 時間グリッドとサンプリング設定の調整が重要
4. **偏光**: 現在はx偏光を使用（z偏光は2次元ベクトル制約により直接使用不可）

## トラブルシューティング

### 励起確率がゼロになる場合
1. 電場振幅を増加（1e10 V/m以上）
2. 共鳴条件の確認（carrier_freq設定）
3. 偏光と双極子軸の組み合わせ確認

### 並列計算エラー
1. MAX_WORKERSを減らす
2. CHUNK_SIZEを小さくする
3. シーケンシャル版（amplitude_duration_2d_sweep_sequential）を使用

### メモリ不足
1. 基底サイズ（V_max）を小さくする
2. 計算点数を減らす
3. sample_strideを大きくする 