## 時間依存シュレディンガー方程式 (TDSE) の数値積分手法のまとめ

### ― Split-Operator 法 & 4 次 Runge–Kutta 法 ―

> **対象式** $i\hbar\dfrac{\partial}{\partial t}\Psi(t)=\hat H(t)\Psi(t)$
> ここで $\hat H(t)=\hat T+\hat V(t)$   （運動エネルギー項＋時間依存ポテンシャル）

---

## 1. Split-Operator (SO) 法

| バリエーション | 分割  | 数値更新式 (Strang 2 次) | 主演算／step | 局所誤差 |
| - | - | - | - | - |
| **(A) 実空間グリッド** | $\hat T=-\dfrac{\hbar^{2}}{2\mu}\nabla^{2}$ <br> $\hat V(q,t)=V_0(q)-\mu(q)\,\mathcal E(t)$ | $\displaystyle \Psi_{t+\Delta t}\!=\!e^{-\tfrac{i}{\hbar}\hat T\frac{\Delta t}{2}}\!e^{-\tfrac{i}{\hbar}\hat V\Delta t}\!e^{-\tfrac{i}{\hbar}\hat T\frac{\Delta t}{2}}\Psi_t$    | FFT → 位相 ×2<br>座標乗算 ×1 | $\mathcal O(\Delta t^{3})$ |
| **(B) 固有状態展開**  | $\mathbf H_0=\mathrm{diag}(E_v)$ <br> $\mathbf V(t)=-\mathbf\mu\,\mathcal E(t)$             | $\displaystyle \mathbf c_{t+\Delta t} = e^{-i\mathbf H_0\frac{\Delta t}{2\hbar}}\,e^{-i\mathbf V(t)\frac{\Delta t}{\hbar}}\,e^{-i\mathbf H_0\frac{\Delta t}{2\hbar}}\mathbf c_t$ | 対角位相 ×2<br>行列指数 ×1     | $\mathcal O(\Delta t^{3})$ |

### 1-A. 実空間アルゴリズム（FFT 版）

```text
# 前計算
k = 2π * FFTfreq(N, dq)
phase_T = exp(-1j * ħ * k**2 * Δt / (4 μ))

for each step:
    ψ ← FFT(ψ)            ; ψ *= phase_T ; ψ ← IFFT(ψ)
    ψ *= exp(-1j * V(q,t) * Δt / ħ)
    ψ ← FFT(ψ)            ; ψ *= phase_T ; ψ ← IFFT(ψ)
```

### 1-B. 固有状態アルゴリズム（係数空間）

```python
# 固有分解 once
eigval, U = np.linalg.eig(mu_matrix)        # μ = U Λ U⁻¹
phase_H0  = np.exp(-1j * E * Δt / (2*ħ))
omega_mu  = -1j * Δt * eigval / ħ           # 前掛け

for each step:
    phase_V = np.exp(omega_mu * E(t))       # diag(e^{-i λ E Δt/ħ})
    c *= phase_H0
    c = U @ (phase_V * (U.conj().T @ c))
    c *= phase_H0
```

*固有分解は 1 回だけ。行列–ベクトル積 2 回／step。*

### 1-C. 高次 (Suzuki 4 次) スプリット

係数
$s_1=\dfrac{1}{2-2^{1/3}},\;s_2=-\dfrac{2^{1/3}}{2-2^{1/3}}$

```
(T/2, V, T/2) with s1
(T/2, V, T/2) with s2
(T/2, V, T/2) with s1        → 局所誤差 O(Δt⁵)
```

FFT 7 回 or 行列指数 7 回／step。

---

## 2. 4 次 Runge–Kutta (RK4) 法

### 2-A. 実空間グリッドでの微分形式

$f(\Psi,t)= -\dfrac{i}{\hbar}\hat H(t)\Psi$

### 2-B. 固有状態展開で *明示的に* $\mathbf H_0$ と電場を用いる RK4

```math
\begin{aligned}
k_1 &= -\tfrac{i}{\hbar}\bigl[\mathbf H_0+\mathbf V(t)\bigr]\,\mathbf c_t\\
k_2 &= -\tfrac{i}{\hbar}\bigl[\mathbf H_0+\mathbf V(t+\tfrac{\Delta t}{2})\bigr]
      \!\bigl(\mathbf c_t+\tfrac{\Delta t}{2}k_1\bigr)\\
k_3 &= -\tfrac{i}{\hbar}\bigl[\mathbf H_0+\mathbf V(t+\tfrac{\Delta t}{2})\bigr]
      \!\bigl(\mathbf c_t+\tfrac{\Delta t}{2}k_2\bigr)\\
k_4 &= -\tfrac{i}{\hbar}\bigl[\mathbf H_0+\mathbf V(t+\Delta t)\bigr]
      \!\bigl(\mathbf c_t+\Delta t\,k_3\bigr)\\[6pt]
\end{aligned}
```
```math
\begin{aligned}
\boxed{\;
\mathbf c_{t+\Delta t}= \mathbf c_t
         +\tfrac{\Delta t}{6}\bigl(k_1+2k_2+2k_3+k_4\bigr)}
\end{aligned}
```

* 行列–ベクトル積 **4 回**／step
* 局所誤差 $\mathcal O(\Delta t^{5})$
* ユニタリ性は数値的に保たれない → 長時間は誤差累積

---

## 3. それぞれの手法の比較

|              | SO (2 次)                             | SO (4 次)                   | RK4 (係数)                   |
| ------------ | ------------------------------------ | -------------------------- | -------------------------- |
| 主演算 / step   | 2 FFT + 1 乗算<br>or 2 MatVec + 1 expm | 7 FFT / 7 expm             | 4 MatVec                   |
| 局所誤差         | $\mathcal O(\Delta t^{3})$           | $\mathcal O(\Delta t^{5})$ | $\mathcal O(\Delta t^{5})$ |
| ユニタリ性        | 近似的に保持 (ノルム◎)                        | 同左                         | 崩れる (要再正規化)                |
| 非線形 V(t) が急変 | △ (指数計算重)                            | △△                         | ◎ (柔軟)                     |
| 状態数 N ≫ 1    | 約 2× 速                               | FFT 多い分やや劣                 | 行列指数不要だが 4 MatVec          |

---

### 4. 使い分けガイド

1. **数 10〜100 状態・長時間・ノルム重視** → **Split-Operator 2 次**
2. **さらに高精度を要す** → **Split-Operator 4 次**
3. **時間に強く依存する外場・偏光が変動** → **RK4 (係数)**
4. **グリッドが必要（トンネル・干渉を描写）** → 実空間 **SO**（FFT）

表を見れば、SO は“位相×指数×位相”の 3 操作で済むため
**行列演算が半減**し、大規模系ほど有利であることが分かります。
