# 厳密 Cartesian split-operator 法の原理

## 1. 「厳密」の意味

このライブラリの

```python
algorithm = "split_operator"
split_interaction = "cartesian"
```

は、RK4 と同じ物理 Hamiltonian

$$
H(t)=H_0-\mu_xE_x(t)-\mu_yE_y(t)
$$

を時間発展させる。

ここでいう「厳密 Cartesian」とは、円偏光や楕円偏光を別の有効模型へ
置き換えず、実際の実数 Cartesian 電場成分 $E_x(t),E_y(t)$ をそのまま
使うという意味である。

ただし、有限刻みの時間発展まで解析的に厳密という意味ではない。時間発展には
2次の Strang 分割を使うため、十分滑らかな問題での誤差は次のようになる。

| 性質 | Cartesian split-operator |
|---|---|
| 時間発展させる Hamiltonian | RK4 と同じ。近似模型への置換なし |
| 1ステップの局所誤差 | $O(\Delta t^3)$ |
| 固定時間までの大域誤差 | $O(\Delta t^2)$ |

## 2. 出発点となる方程式

内部の Hamiltonian は `rad/fs` で表現され、$\hbar$ は単位変換後の式へ
吸収されている。状態ベクトルは

$$
\frac{d\psi}{dt}=-iH(t)\psi(t)
$$

に従う。Hamiltonian を

$$
H(t)=H_0+V(t),
\qquad
V(t)=-\mu_xE_x(t)-\mu_yE_y(t)
$$

と分ける。この符号はライブラリ全体で採用している $-\mu E$ 規約である。

作業基底では $H_0$ は実対角行列である。

$$
H_0=\operatorname{diag}(\omega_1,\ldots,\omega_N)
$$

したがって、自由発展の半ステップは各係数への位相乗算だけで計算できる。

$$
\left[e^{-iH_0\Delta t/2}\psi\right]_n
=e^{-i\omega_n\Delta t/2}\psi_n
$$

## 3. 複素 Jones ベクトルを直接使わない理由

円偏光は、たとえば

$$
p=\frac{1}{\sqrt{2}}(1,+i)
$$

という複素 Jones ベクトルで表せる。しかし $p$ は瞬時電場そのものではなく、
成分間の振幅比と位相差を表す複素表現である。物理的な電場は実数であり、
複素解析信号を使っても最後に実部を取る。

$$
\boldsymbol{E}(t)=\operatorname{Re}[\mathcal{E}(t)p]
=\bigl(E_x(t),E_y(t)\bigr)
$$

$\mathcal{E}(t)=F(t)e^{i\theta(t)}$ なら、上の Jones ベクトルから

$$
E_x(t)=\frac{F(t)}{\sqrt{2}}\cos\theta(t),
\qquad
E_y(t)=-\frac{F(t)}{\sqrt{2}}\sin\theta(t)
$$

という実数成分が得られる。

一方、複素係数をそのまま用いた

$$
-p_x\mu_x-p_y\mu_y
$$

は一般には Hermitian ではない。$\mu_x,\mu_y$ が Hermitian でも、係数が
複素数なら、その線形結合が Hermitian である保証はないからである。

厳密 Cartesian モードは、非 Hermitian 行列を後から対称化しない。Jones
ベクトルから構成済みの実電場 $E_x(t),E_y(t)$ を使うため、瞬時相互作用は
最初から Hermitian になる。

$$
V(t)^\dagger
=-\mu_x^\dagger E_x(t)-\mu_y^\dagger E_y(t)
=V(t)
$$

これが今回の方法の最も重要な点である。

## 4. Strang 分割と電場の中点値

時刻 $t_k$ から $t_k+\Delta t$ までの更新を

$$
\psi(t_k+\Delta t)\approx
e^{-iH_0\Delta t/2}
e^{-iV(t_k+\Delta t/2)\Delta t}
e^{-iH_0\Delta t/2}\psi(t_k)
$$

と近似する。

このライブラリでは、隣接電場サンプルの間隔 `field_dt` と伝播刻みの間に

$$
\Delta t=\text{propagation\_dt}=2\,\text{field\_dt}
$$

という契約がある。長さ $2N_{\mathrm{step}}+1$ の電場配列のうち、添字
`1, 3, 5, ...` が各伝播ステップの中点値になる。

## 5. 電場を振幅と方向へ分ける

各中点で実電場を

$$
A_k=\sqrt{E_{x,k}^2+E_{y,k}^2},
\qquad
\phi_k=\operatorname{atan2}(E_{y,k},E_{x,k})
$$

と定義する。すると

$$
E_{x,k}=A_k\cos\phi_k,
\qquad
E_{y,k}=A_k\sin\phi_k
$$

であり、相互作用は

$$
V_k=-A_k\mu_{\phi_k},
\qquad
\mu_\phi=\cos\phi\,\mu_x+\sin\phi\,\mu_y
$$

と書ける。$A_k=0$ のとき方向は意味を持たないため、実装は相互作用を
恒等演算として扱い、$\phi_k$ を使用しない。

## 6. M 位相回転が核心になる理由

M を明示した線形分子基底を $|n\rangle=|v,J,M\rangle$ とし、各基底状態の
磁気量子数を $M_n$ と書く。対角ユニタリ行列

$$
D(\phi)_{nn}=e^{iM_n\phi}
$$

を定義する。現在の LinMol の Cartesian 双極子規約では

$$
\boxed{D(\phi)\mu_xD(\phi)^\dagger
=\cos\phi\,\mu_x+\sin\phi\,\mu_y=\mu_\phi}
$$

が成り立つ。行列要素ごとには

$$
\left[D(\phi)\mu_xD(\phi)^\dagger\right]_{nm}
=e^{i(M_n-M_m)\phi}(\mu_x)_{nm}
$$

である。選択則 $\Delta M=\pm1$ を持つ要素へ回転方向に対応した位相が付くため、
$x$ 成分が任意角度の Cartesian 成分へ回転する。この恒等式により、時間ごとに
変化する $\mu_\phi$ を毎回対角化せず、固定した $\mu_x$ と安価な対角位相
$D(\phi)$ だけで表せる。

## 7. 相互作用指数の高速な計算

$\mu_x$ は Hermitian なので、伝播開始時に1回だけ

$$
\mu_x=Q\Lambda Q^\dagger
$$

と固有値分解できる。$\Lambda$ は実対角行列、$Q$ はユニタリ行列である。
回転恒等式から

$$
\mu_{\phi_k}=D(\phi_k)Q\Lambda Q^\dagger D(\phi_k)^\dagger
$$

となる。$V_k=-A_k\mu_{\phi_k}$ なので

$$
\boxed{e^{-iV_k\Delta t}
=D(\phi_k)Qe^{+iA_k\Lambda\Delta t}Q^\dagger D(\phi_k)^\dagger}
$$

を得る。中央の指数は対角なので、各固有成分へ
$e^{+iA_k\lambda_j\Delta t}$ を掛けるだけでよい。符号が $+i$ なのは
相互作用が $V=-\mu E$ だからである。

## 8. 1ステップの処理

```text
# 前計算：伝播呼び出しごとに1回
mu_x = Q diag(lambda) Q^dagger
half_phase[n] = exp(-i H0[n] dt / 2)

# 各ステップ
psi <- half_phase * psi
A   <- hypot(Ex_mid, Ey_mid)
phi <- atan2(Ey_mid, Ex_mid)

psi[n] <- exp(-i M[n] phi) * psi[n]  # D(phi)^dagger
c      <- Q^dagger @ psi
c[j]   <- exp(+i dt A lambda[j]) * c[j]
psi    <- Q @ c
psi[n] <- exp(+i M[n] phi) * psi[n]  # D(phi)

psi <- half_phase * psi
```

全体では

$$
\psi_{k+1}\approx e^{-iH_0\Delta t/2}
D_kQe^{+iA_k\Lambda\Delta t}Q^\dagger D_k^\dagger
e^{-iH_0\Delta t/2}\psi_k
$$

となる。コード中の `U` は $Q$、`eigvals` は $\Lambda$ の対角要素、
`magnetic_quantum_numbers` は $M_n$ に対応する。

## 9. Hermitian 性とノルム保存

必要な条件は次のとおりである。

1. $H_0$ が実対角である。
2. $\mu_x,\mu_y$ が Hermitian である。
3. $E_x(t),E_y(t)$ が実数である。
4. $D(\phi)$ と双極子行列の回転共変性が成り立つ。

このとき、自由発展位相、$D(\phi)$、$Q$、相互作用の固有値位相はすべて
ユニタリである。1ステップはユニタリ行列の積なので、丸め誤差を除けば

$$
\|\psi_{k+1}\|_2=\|\psi_k\|_2
$$

を保つ。

実装は Hermitian でない入力を $(M+M^\dagger)/2$ で暗黙修復しない。
$\mu_x,\mu_y$ の Hermitian 性をスケール依存の丸め誤差範囲で検証し、
範囲を超える場合はエラーにする。

通常は `renorm=False` のままでよい。`renorm=True` は明示的な追加操作であり、
split-operator の原理上必要な処理ではない。

## 10. 毎時刻の対角化より速い理由

素朴な方法では、中点ごとに

$$
V_k=-\mu_xE_{x,k}-\mu_yE_{y,k}
$$

を作り、その指数を得るため毎回固有値分解する。密行列の固有値分解は概ね
$O(N^3)$ なので、時間ステップ数が多いと高価になる。

M 位相回転法では次の計算量になる。

- $\mu_x$ の固有値分解：伝播呼び出しごとに1回、$O(N^3)$
- 各ステップの2回の密行列–ベクトル積：$O(N^2)$
- M位相と固有値位相の要素積：$O(N)$

時間方向に繰り返される $O(N^3)$ を除去することが高速化の核心である。
RK4 は1ステップ内で複数回 Hamiltonian を状態へ作用させるが、この split 法の
相互作用部分は主に $Q^\dagger\psi$ と $Q\psi$ の2回の密 matrix-vector 積で済む。

現在の CPU ベンチマークでは、検証と最初の固有値分解を含む公開API全体で

| 次元 | Cartesian split の dense RK4 比速度 |
|---:|---:|
| 32 | 約 1.67 倍 |
| 72 | 約 2.31 倍 |

だった。速度比はハードウェア、BLAS、状態数、ステップ数に依存する。

RK4 は4次、現在の split 法は2次である。同じ刻みで速いことと、同じ誤差まで
収束させたとき常に速いことは同義ではないため、実計算では速度と刻み収束を
両方確認する必要がある。

## 11. 固定直線偏光はさらに単純になる

電場方向が時間を通して固定されているなら

$$
\boldsymbol{E}(t)=s(t)\boldsymbol{u},
\qquad \|\boldsymbol{u}\|=1
$$

と分解できる。$s(t)$ は符号を持つ実スカラーである。相互作用を

$$
V(t)=s(t)A_{\boldsymbol{u}},
\qquad
A_{\boldsymbol{u}}=-u_x\mu_x-u_y\mu_y
$$

と書き、$A_{\boldsymbol{u}}$ を1回だけ対角化する。この経路は方向が
変化しないため、M ラベルも M 位相回転も必要としない。

## 12. `helicity_projected` との違い

2つのモードは計算手順だけでなく、表す物理模型が異なる。

| 項目 | `cartesian` | `helicity_projected` |
|---|---|---|
| 位置づけ | 既定の厳密 Cartesian 模型 | 明示的な近似模型 |
| 入力場 | 実数の $E_x(t),E_y(t)$ | 複素 Jones ベクトルと実スカラー波形 |
| Hamiltonian | $H_0-\mu_xE_x-\mu_yE_y$ | 片方向遷移 $T$ と $T^\dagger$ から構成 |
| 円・楕円偏光 | 実 Cartesian 成分をそのまま扱う | helicity 選択を上三角成分へ投影 |
| 強場・超短パルス | 元の Cartesian Hamiltonian を維持 | Cartesian 結果と異なり得る |
| 基底順序への依存 | M回転共変性に依存 | 上三角を片方向と読む基底順序に依存 |

projected モードは

$$
T=\operatorname{triu}(-p_x\mu_x-p_y\mu_y,k=1),
\qquad
A_{\mathrm{projected}}=T+T^\dagger
$$

を使う。これは Hermitian だが、元の複素線形結合と数学的に同じ演算子ではない。
許可する遷移方向を選んだ RWA 的な近似模型である。したがって projected と
Cartesian の結果差は、基本的には数値誤差ではなく模型差として解釈する。

## 13. 適用条件とエラーになる場合

共通条件：

- $H_0$ は実対角であること。
- $\mu_x,\mu_y$ は有限な Hermitian 行列であること。
- 初期状態、演算子、電場の次元が一致すること。
- `dt > 0` であること。
- 電場配列長は $2N_{\mathrm{step}}+1$ の奇数であること。

方向が変化する Cartesian 場の追加条件：

- 各基底状態に対応する M 配列があること。
- M 配列長が Hilbert 空間の次元と一致すること。
- $\mu_x,\mu_y$ が M 回転共変性を満たすこと。

実装は少なくとも $\phi=\pi/2$ で

$$
D(\pi/2)\mu_xD(\pi/2)^\dagger\approx\mu_y
$$

を丸め誤差スケールで検証する。LinMol の物理テストでは複数角度に対して
完全な回転恒等式を確認している。

この条件を満たさない一般の多準位模型には、M 位相回転による高速化をそのまま
適用できない。その場合は RK4、固定方向 split、または模型固有の回転生成子が
必要になる。

入力が疎行列でも、固有ベクトル $Q$ は一般に密行列になる。現在の split 法は
入力を密化してスペクトル分解するため、`sparse=True` は split 法の疎行列
メモリスケーリングを意味しない。

## 14. 精度の確認方法

### 14.1 ノルムを確認する

`renorm=False` のまま

$$
\left|\|\psi(t)\|_2-1\right|
$$

を確認する。大きなドリフトがある場合は刻みだけでなく、入力演算子の
Hermitian 性や非有限値も疑う。

### 14.2 刻み収束を確認する

$\Delta t$ と $\Delta t/2$ で計算し、十分細かい領域で誤差が約1/4になるかを
確認する。現在の円偏光テストでは、Cartesian split と RK4 の差について

$$
\frac{\epsilon(\Delta t)}{\epsilon(\Delta t/2)}\approx3.987
$$

が得られ、2次収束と整合している。

### 14.3 RK4と比較する

Cartesian モードは同じ Hamiltonian を使うので、刻みを細かくすれば RK4 と
同じ解へ収束する。一方、`helicity_projected` は異なる模型なので、RK4との
差が刻みとともにゼロになることを要求してはいけない。

## 15. 使用方法

高水準APIでは次のように指定する。

```python
from rovibrational_excitation.core.propagation import SchrodingerPropagator

propagator = SchrodingerPropagator(
    algorithm="split_operator",
    split_interaction="cartesian",  # 既定値
    backend="numpy",
    renorm=False,
)

psi_t = propagator.propagate(
    hamiltonian,
    electric_field,
    dipole_matrix,
    initial_state,
)
```

シミュレーションパラメータでは

```python
algorithm = "split_operator"
split_interaction = "cartesian"
```

と指定する。LinMol の円偏光・楕円偏光・時間変化する xy 方向には
`use_M=True` が必要である。高水準APIが
`dipole_matrix.basis.M_array` を低水準カーネルへ渡す。

## 16. よくある疑問

### 円偏光なら Hamiltonian は非 Hermitian にならないのか

ならない。複素 Jones ベクトルは偏光の表現であり、実際に Hamiltonian へ入る
$E_x(t),E_y(t)$ は実数である。実数係数と Hermitian な $\mu_x,\mu_y$ の
組合せなので、Cartesian Hamiltonian は Hermitian である。

### $\mu_x+i\mu_y$ をそのまま使ってはいけないのか

$\mu_x\pm i\mu_y$ は $\Delta M$ を選ぶ球面テンソル成分として有用だが、
片方だけでは一般に Hermitian ではない。吸収側と放出側を分ける近似模型を
作るなら利用できるが、それが `helicity_projected` であり、実 Cartesian
Hamiltonian と同一ではない。

### 角度が $+\pi$ から $-\pi$ へ飛んでも大丈夫か

`atan2` の角度表示は不連続に見えるが、整数の M に対する
$D(\phi)=e^{iM\phi}$ と、それが表す $\mu_\phi$ は $2\pi$ 周期である。
したがって、物理的な相互作用はこの表示上の折り返しで変化しない。

### split-operator は常に RK4 より速いのか

常にではない。状態数、ステップ数、BLAS、要求精度で変わる。現在の実装では
毎ステップの対角化を除き、相互作用を2回の dense matvec へ落とせることが
速度上の利点である。一方、2次法なので要求精度によってはより細かい刻みが要る。

### GPUでも同じ原理か

CPU と CuPy 経路は同じ M 位相回転と相互作用指数を実装している。ただし現在の
検証環境には CUDA 実機がなく、GPU 数値一致と性能は未検証である。

## 17. 実装・テスト・ベンチマーク

主要実装：

- `src/rovibrational_excitation/core/propagation/algorithms/split_operator/schrodinger.py`
  - `_propagate_rotating_xy_numpy`: CPU の M 位相回転カーネル
  - `_splitop_rotating_xy_cupy`: GPU の対応カーネル
  - `_validate_xy_rotation_covariance`: 回転共変性の検証
  - `splitop_schrodinger`: 固定方向・回転方向・projected の振り分け
- `src/rovibrational_excitation/core/propagation/schrodinger.py`
  - 高水準APIから実電場成分と M 配列を渡す

物理テスト：

- `tests/physics/test_split_operator_polarization.py`
  - LinMol 双極子の M 回転共変性
  - 円偏光 Cartesian split の RK4 への2次収束
  - projected モードの $\Delta M$ 選択
  - 非 Hermitian 双極子と未規格化 Jones ベクトルの拒否
  - 高水準APIを通した Cartesian / RK4 / projected の比較

速度・精度記録：

- `benchmarks/run_split_operator.py`
- `benchmarks/split-polarization-v0.3.json`
- `benchmarks/README.md`

## 18. 一文でまとめると

複素偏光を非 Hermitian な複素双極子行列として指数化するのではなく、実電場の
瞬時方向を M 依存の対角位相回転へ置き換えることで、RK4と同じ Hermitian
Hamiltonian を保ったまま、相互作用行列の固有値分解を1回に減らす方法である。
