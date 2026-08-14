# Propagation module

量子状態の時間発展を行う高水準境界です。公開 API は純粋状態、密度行列、規格化済み統計重みを持つインコヒーレント ensemble を明示的に区別します。

## 必須の計算設定

公開 `propagate()` は `PropagationOptions` と coupling 指定を必須にします。algorithm、backend、dense/CSR、trajectory、stride、次元化、逐次規格化は推測されません。

```python
from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.core.propagation import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)
from rovibrational_excitation.core.propagation.capabilities import (
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

## 純粋状態

```python
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.states import PureState

solver = SchrodingerPropagator(
    algorithm="rk4",
    backend="numpy",
    sparse=False,
    renorm=False,
    validate_units=True,
)

state_or_pair = solver.propagate(
    hamiltonian=hamiltonian,
    efield=field,
    dipole_matrix=dipole,
    initial_state=PureState(amplitudes),
    options=options,
    coupling_mode="cartesian",
    axes="xy",
    return_times=True,
)
```

scalar coupling では `axes` を渡さず、`coupling_axis="x"` などを必須指定します。Cartesian coupling へ `coupling_axis` を渡すこと、および scalar coupling へ `axes` を渡すことはエラーです。

## 密度行列

```python
from rovibrational_excitation.core.propagation import LiouvillePropagator
from rovibrational_excitation.core.states import DensityState

solver = LiouvillePropagator(backend="numpy", validate_units=True)
time_fs, rho = solver.propagate(
    hamiltonian=hamiltonian,
    efield=field,
    dipole_matrix=dipole,
    initial_state=DensityState(rho0),
    options=options,
    coupling_mode="cartesian",
    axes="xy",
    return_times=True,
)
```

Liouville 経路は NumPy dense RK4、renormalization disabled のみを受け付けます。`DensityState` は有限、正方、Hermitian、positive semidefinite、trace one でなければなりません。入力は自動修復されません。

## インコヒーレント ensemble

```python
from rovibrational_excitation.core.propagation import MixedStatePropagator
from rovibrational_excitation.core.states import IncoherentEnsemble

ensemble = IncoherentEnsemble(component_amplitudes)
solver = MixedStatePropagator(
    algorithm="rk4",
    backend="numpy",
    sparse=False,
    renorm=False,
    validate_units=True,
)
rho = solver.propagate(
    hamiltonian=hamiltonian,
    efield=field,
    dipole_matrix=dipole,
    initial_state=ensemble,
    options=options,
    coupling_mode="scalar",
    coupling_axis="z",
)
```

各入力ベクトルのノルム二乗が生の統計重みです。`IncoherentEnsemble` が重みを規格化した後、各純粋成分を別々に伝播し、`sum_i w_i |psi_i><psi_i|` を計算します。成分間のコヒーレント交差項は作りません。

## Split operator の相互作用モード

`split_operator` を選ぶ場合、solver constructor と公開 `propagate()` の両方で同じ `split_interaction` を明示します。

- `cartesian`: RK4 と同じ Hamiltonian を使う物理参照経路
- `helicity_projected`: 選択則で射影する明示的な近似経路

省略、constructor との不一致、RK4 への指定はすべて計算前にエラーになります。

## 時間と戻り値

時間刻みは `ElectricField` の grid だけから決まります。solver-level `dt`、automatic timestep、accuracy による暗黙調整はありません。

P2.5 完了までは `return_times=True` により `(time, state)`、`False` により state array を返す移行 API です。trajectory と stride は `PropagationOptions` が一意に所有します。非公開 `_propagate_array()` は既存 optimizer と数値 kernel の移行用であり、新規コードの公開 API として使用しません。
