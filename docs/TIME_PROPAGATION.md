# 時間発展の数値契約

この文書は開発版 `0.3.0.dev1` の公開時間発展境界を説明する。通常シミュレーションの
設定項目は [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)、Cartesian
split-operator の導出は
[CARTESIAN_SPLIT_OPERATOR.md](CARTESIAN_SPLIT_OPERATOR.md) を参照する。

## 1. 共通の物理式

Schrödinger 経路と Liouville–von Neumann 経路はいずれも

$$
H(t)=H_0-\mu E(t)
$$

を使う。Cartesian coupling では

$$
H(t)=H_0-\mu_xE_x(t)-\mu_yE_y(t)
$$

である。符号、軸、偏光、規格化を時間発展器が推測または修復することはない。

内部の伝播配列は変換済みの角周波数系で扱い、時間は fs で管理する。公開入力に
必要な単位と変換境界は [UNIT_SYSTEM.md](UNIT_SYSTEM.md) を参照する。

## 2. 時間格子

`TimeGrid` は等間隔・狭義単調増加・奇数長の field grid を要求する。
伝播1ステップには左端・中点・右端の3つの電場サンプルを使うため、

$$
\mathtt{propagation\_dt\_fs}=2\,\mathtt{field\_dt\_fs}
$$

であり、$N$ ステップには $2N+1$ 個の field sample が必要である。

`TimeGrid.from_bounds(t_start_fs, t_end_fs, field_dt_fs)` は端点を丸めたり延長
したりしない。`t_end_fs - t_start_fs` が `2 * field_dt_fs` の整数倍で
なければエラーになる。電場の `tlist` は `TimeGrid.field_times_fs` と配列として
完全一致しなければならず、resample は行わない。

## 3. 必須の型付き入力

1回の計算は次の3つを明示して構成する。

- `PropagationProblem`: model、field、`TimeGrid`、初期状態
- `PropagationOptions`: algorithm、backend/storage、trajectory、stride、
  scaling、renormalization
- propagator: 初期状態の意味に対応した solver

`ExecutionPolicy` は backend を `numpy` または `cupy`、matrix storage を
`dense` または `csr` として別々に指定する。未知の値や非対応の組合せから別経路へ
fallback しない。

## 4. 対応する数値経路

| 初期状態 | algorithm | backend | storage | 現状 |
|---|---|---|---|---|
| pure state | RK4 | NumPy | dense | 対応・CPU 検証済み |
| pure state | RK4 | NumPy | CSR | 対応・Numba CSR 検証済み |
| pure state | split operator | NumPy | dense | 対応・CPU 検証済み |
| pure state | split operator | NumPy | CSR | 入力可能。ただし固有値分解前に dense 化 |
| incoherent ensemble | RK4 / split operator | NumPy | dense / CSR | 各 pure component を別々に伝播して密度演算子を重み付き加算 |
| density matrix | RK4 | NumPy | dense | 対応。Liouville–von Neumann 経路 |
| density matrix | split operator | 任意 | 任意 | 非対応・エラー |
| density matrix | RK4 | CuPy または CSR | 任意 | 非対応・エラー |
| pure state / incoherent ensemble | RK4 / split operator | CuPy | dense | 実 CUDA 数値受入れ済み（D-159） |
| 任意 | 任意 | CuPy | CSR | 非対応・エラー |

Production SymTop はさらに狭く、NumPy dense/CSR RK4 のみを受け付ける。
split operator、CuPy、all-isomer pure state、optimization は明示的に拒否する。

## 5. RK4

状態ベクトルでは

$$
\dot\psi(t)=-iH(t)\psi(t)
$$

を古典的4次 Runge–Kutta で更新する。1ステップ内の $H$ は field grid の左端、
中点、中点、右端から構成する。したがって `field_dt_fs` を伝播刻みとして使っては
ならず、実際の展開係数の更新幅はその2倍である。

NumPy dense は dense 行列–ベクトル積を使う。NumPy CSR は operator を一度 CSR
表現へ変換し、Numba 内の CSR matvec と再利用する作業配列で同じ RK4 stage 順序を
実行する。`sparse` は近似や threshold pruning を意味しない。

RK4 は厳密にはユニタリではない。`RenormalizationPolicy.PER_STEP` を指定した場合
だけ各ステップで規格化し、`DISABLED` のとき暗黙に補正しない。

## 6. Split operator

実装されているのは係数空間の2次 Strang split である。

$$
\psi_{k+1}\approx
e^{-iH_0\Delta t/2}
e^{-iV(t_k+\Delta t/2)\Delta t}
e^{-iH_0\Delta t/2}\psi_k .
$$

`H0` は実対角、相互作用演算子は Hermitian でなければならない。実装は非 Hermitian
入力を対称化せず、scale-aware な丸め許容値を超えた場合はエラーにする。

公開呼び出しでは `split_interaction` を毎回明示し、propagator constructor と同じ
値を渡す。省略または不一致はエラーになる。

### `cartesian`（標準・厳密な物理 Hamiltonian）

実数の $E_x(t),E_y(t)$ と Hermitian な $\mu_x,\mu_y$ を使い、RK4 と同じ
Cartesian Hamiltonian を伝播する。

- field direction が固定なら、固定方向の相互作用を1回だけ対角化する。
- M-resolved LinMol で xy direction が変わる場合は、$M$ 位相回転を用いる。
- 変動方向で M label または xy 回転共変性がない場合はエラーにする。

ここで「厳密」は Hamiltonian を helicity 有効模型へ置換しないという意味であり、
有限刻みの Strang 誤差がないという意味ではない。局所誤差は
$O(\Delta t^3)$、固定時間までの大域誤差は $O(\Delta t^2)$ である。

### `helicity_projected`（明示的な近似）

複素 Jones vector から一方向遷移 operator の上三角成分を取り、その随伴を加えて
Hermitian な有効相互作用を構成する。selection-rule を保った軽量な近似であり、
強電場・超短パルスでは Cartesian 経路と一致する保証はない。使用者が
`split_interaction="helicity_projected"` を明示した場合だけ選択される。

CSR operator を渡すことはできるが、split operator は spectral decomposition のため
dense 化する。したがって現状の `storage="csr"` は split のメモリ・計算量を疎行列
のまま保つ機能ではない。

実空間 FFT split、4次 Suzuki split、adaptive split は実装していない。

## 7. 初期状態ごとの意味

- `PureState`: 1本の波動関数を伝播する。
- `IncoherentEnsemble`: 規格化済み統計重みごとに pure state を独立伝播し、
  $\rho(t)=\sum_i w_i|\psi_i(t)\rangle\langle\psi_i(t)|$ を返す。
- `DensityState`: trace one、有限、正方、Hermitian、positive-semidefinite の
  density matrix を Liouville 経路へ渡す。入力を規格化・対称化・clip しない。

複数の normal-simulation `initial_states` は incoherent ensemble ではなく、
等振幅・同位相の coherent superposition である。incoherent sum が必要なら
`IncoherentEnsemble` と `MixedStatePropagator` を使う。

## 8. trajectory と stride

型付き `PropagationResult` は、trajectory を要求した場合に開始点と正確な終了点を
必ず含む。`sample_stride` は積分刻みや field index を変更せず、計算済み trajectory
の出力間引きだけを指定する。step 数が stride で割り切れない場合、最後の出力区間
だけが短くなる。

`return_trajectory=False` は終了時刻1点と final state を返す。結果の state は
backend-native で保持する契約であり、host copy が必要なら
`PropagationResult.to_numpy()` を明示的に呼ぶ。

## 9. scaling、backward、convergence

`ScalingMode.DIMENSIONAL` と `NONDIMENSIONAL` は必須の選択である。無次元化を
選んでも公開時刻は fs で返し、使用した scale は result metadata に記録する。

公開 backward propagation は現在 NumPy・RK4・dimensional に限る。field sample を
逆順に使い、負の伝播刻みで積分する。他の backend、split、nondimensional との
組合せはエラーになる。

convergence check は opt-in である。呼出側が coarse/fine の2つの格子、observable、
tolerance を明示し、最大絶対差を比較する。solver が刻みを自動変更、丸め、resample
することはない。

## 10. CUDA の扱い

CuPy を要求して利用できない場合は即座にエラーにし、NumPy へ fallback しない。
低レベル Schrödinger RK4 は CPU と同じ `H0 - mu E` の4段計算を、split は
固定Cartesian・M回転Cartesian・helicity-projected の既存式をCuPy上で実行する。
どちらも trajectory、stride、renormalization を保持してCuPy配列を直接返し、状態・
trajectory配列をhostへ移さない。分岐と既存エラーを守るscalar同期は残る。実 CUDA 数値受入れでは、
RTX 5070 Ti上のcleanな`b9de848`で全15件のGPU testと5経路のschema-v1証跡が成功し、
backend identity・転送量・同期済み時間を記録した。32状態の時間は速度保証ではない。
最終release候補ではself-hosted workflowを再実行する。GPU test がskipされたことだけを
CUDA対応の検証根拠にはしない。

## 11. 参照先

- 公開入力型: `core.time`、`core.execution`、`dynamics.problem`、
  `dynamics.options`、`dynamics.result`
- capability check: `dynamics.capabilities`
- RK4: `dynamics.algorithms.rk4`
- split operator: `dynamics.algorithms.split_operator`
- 権威となる不変条件:
  [PHYSICS_CONTRACTS.md](refactoring/PHYSICS_CONTRACTS.md)
- 受理済み判断: [DECISIONS.md](refactoring/DECISIONS.md)
