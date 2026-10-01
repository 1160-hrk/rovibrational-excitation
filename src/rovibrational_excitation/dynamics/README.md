# Propagation module

量子状態の時間発展を行う高水準境界です。公開 API は純粋状態、密度行列、規格化済み統計重みを持つインコヒーレント ensemble を明示的に区別します。

## 必須の計算設定

公開 `propagate()` は `PropagationOptions` と coupling 指定を必須にします。algorithm、backend、dense/CSR、trajectory、stride、次元化、逐次規格化は推測されません。

密度行列のNumPy/Numba RK4カーネルは、隣接ステップで同じ電場標本を
参照する右端・次の左端のハミルトニアンだけを再利用します。場の添字、
交換子、RK4段階、stride、出力形状は変更しません。

```python
from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.dynamics import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
)

options = PropagationOptions(
    algorithm=PropagationAlgorithm.RK4,
    execution=ExecutionPolicy(
        backend=ArrayBackend.NUMPY,
        storage=MatrixStorage.DENSE,
    ),
    return_trajectory=True,
    sample_stride=1,
    scaling=ScalingMode.DIMENSIONAL,
    renormalization=RenormalizationPolicy.DISABLED,
)
```

solver constructor の algorithm、backend、storage、renormalization と `options` が一致しない場合は、計算前にエラーになります。

## 完全な問題と純粋状態

```python
import numpy as np

from rovibrational_excitation.dynamics import (
    Axis,
    CouplingSpec,
    PropagationProblem,
    SchrodingerPropagator,
    SystemModel,
)
from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.core.time import TimeGrid

time_grid = TimeGrid(np.asarray(field.tlist))
model = SystemModel(
    name="my-model",
    basis=basis,
    hamiltonian=hamiltonian,
    dipole=dipole,
    coupling=CouplingSpec.cartesian("xy"),
    metadata={},
)
problem = PropagationProblem(
    model=model,
    field=field,
    time_grid=time_grid,
    initial_state=PureState(amplitudes),
)
solver = SchrodingerPropagator(
    algorithm="rk4",
    backend="numpy",
    sparse=False,
    renorm=False,
    validate_units=True,
)
result = solver.propagate(problem, options=options)
time_fs = result.times_fs
psi = result.state
```

結合モードと軸は `SystemModel.coupling` が一意に所有します。scalar coupling では `CouplingSpec.scalar(Axis.Z)`、Cartesian coupling では `CouplingSpec.cartesian("xy")` のように指定します。モデルと solver 呼び出し側で軸を二重管理しません。

## 密度行列

```python
from rovibrational_excitation.dynamics import LiouvillePropagator
from rovibrational_excitation.core.states import DensityState

density_problem = PropagationProblem(
    model=model,
    field=field,
    time_grid=time_grid,
    initial_state=DensityState(rho0),
)
solver = LiouvillePropagator(backend="numpy", validate_units=True)
rho_result = solver.propagate(density_problem, options=options)
rho = rho_result.state
```

Liouville 経路は NumPy dense RK4、renormalization disabled のみを受け付けます。`DensityState` は有限、正方、Hermitian、positive semidefinite、trace one でなければなりません。入力は自動修復されません。Numba kernel に渡す直前だけ、同じ値を持つ writable C-order 作業配列を作ります。
内部では `rk4/lvne.py` が検証と数値配列準備を所有し、`rk4/liouville_numpy.py` は検証済み配列だけを受け取る NumPy/Numba 数値カーネルです。後者は単位変換やモデル、設定、I/Oに依存しません。

## インコヒーレント ensemble

```python
from rovibrational_excitation.dynamics import MixedStatePropagator
from rovibrational_excitation.core.states import IncoherentEnsemble

scalar_model = SystemModel(
    name="my-scalar-model",
    basis=basis,
    hamiltonian=hamiltonian,
    dipole=dipole,
    coupling=CouplingSpec.scalar(Axis.Z),
    metadata={},
)
ensemble_problem = PropagationProblem(
    model=scalar_model,
    field=field,
    time_grid=time_grid,
    initial_state=IncoherentEnsemble(component_amplitudes),
)
solver = MixedStatePropagator(
    algorithm="rk4",
    backend="numpy",
    sparse=False,
    renorm=False,
    validate_units=True,
)
rho_result = solver.propagate(ensemble_problem, options=options)
```

各入力ベクトルのノルム二乗が生の統計重みです。`IncoherentEnsemble` が重みを規格化した後、各純粋成分を別々に伝播し、`sum_i w_i |psi_i><psi_i|` を計算します。成分間のコヒーレント交差項は作りません。

## Split operator の相互作用モード

`split_operator` を選ぶ場合、solver constructor と公開 `propagate()` の両方で同じ `split_interaction` を明示します。

- `cartesian`: RK4 と同じ Hamiltonian を使う物理参照経路
- `helicity_projected`: 選択則で射影する明示的な近似経路

省略、constructor との不一致、RK4 への指定はすべて計算前にエラーになります。

## 時間と戻り値

時間刻みは `ElectricField` の grid だけから決まります。solver-level `dt`、automatic timestep、accuracy による暗黙調整はありません。

公開 `propagate()` は常に `PropagationResult` を返します。`return_times` は廃止済みです。最終状態だけを要求した場合も `times_fs` は終端時刻1点を持ちます。trajectory は厳密な開始点と終端を含み、stride が step 数を割り切らない場合は計算済み終端だけを出力へ追加します。積分刻み、電場サンプル、状態更新は変えません。

`state` は選択した backend 上に残ります。NumPy が必要な保存・解析境界では `host_result = result.to_numpy()` と明示します。metadata は schema/package version、モデル宣言、全実行選択、時間格子、無次元化時に実際に使った scale、設定 hash を持つ読み取り専用JSON値です。設定 hash は宣言済み契約を対象とし、Hamiltonian・dipole・field の数値配列本体は暗黙にhostへ転送してhashしません。

このbackend-native方針は公開結果境界では成立していますが、現行CuPy
RK4/split低水準アダプタにはhost往復が残っています。実GPUで移行と
CPU/GPU一致を検証するまでは、Phase 5の未完了事項として扱います。

非公開 `_propagate_array()` は既存 optimizer と数値 kernel の移行用です。新規コードの公開 API として使用しません。
