# `fla/ops/kda` 目录分析

> 来源仓库：[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)

---

## 一、目录结构总览

```
fla/ops/kda/
├── __init__.py                      # 包入口，对外暴露两个主函数
├── naive.py                         # 纯 PyTorch 参考实现，用于正确性验证
├── gate.py                          # KDA 遗忘门（forget gate）的前向/反向计算
├── wy_fast.py                       # WY 表示的快速 Triton 计算（intra-chunk w/u）
├── chunk_intra_token_parallel.py    # intra-chunk 注意力的 token 并行实现
├── chunk_intra.py                   # intra-chunk 完整前向/反向（融合了 token 并行）
├── chunk_fwd.py                     # 分块前向主流程编排
├── chunk_bwd.py                     # 分块反向主流程编排
├── chunk.py                         # 对外 API：ChunkKDAFunction + chunk_kda
├── fused_recurrent.py               # 融合递归（sequential）Triton 实现
└── backends/
    ├── __init__.py                  # 后端注册表（FlashKDA + TileLang）
    ├── flashkda.py                  # MoonshotAI CUTLASS 推理后端
    └── tilelang/
        ├── __init__.py              # TileLang 后端类
        └── chunk_bwd_dqkg.py        # TileLang 反向 dq/dk/dg 实现
```

---

## 二、各文件职责与函数说明

### 1. `__init__.py`（包入口）

**职责**：对外导出 `fla.ops.kda` 包的公共 API。

| 导出符号 | 来源文件 | 说明 |
|---|---|---|
| `chunk_kda` | `chunk.py` | 主力训练/推理 API，分块计算 KDA |
| `fused_recurrent_kda` | `fused_recurrent.py` | 顺序递归 API，适用于推理或短序列 |

---

### 2. `naive.py`（参考实现）

**职责**：提供纯 PyTorch 的朴素实现，用于和高效 Triton 实现做数值对比，也便于理解算法逻辑。

#### 函数

**`naive_recurrent_kda(q, k, v, g, beta, scale, initial_state, output_final_state)`**

- **原理**：逐 token 递推。状态矩阵 `S` 每步先乘以遗忘门 `exp(g_i)`，再做 delta-rule 更新 `S += beta_i * k_i ⊗ (v_i - k_i^T S)`，最后 `o_i = q_i^T S`。
- **支持 GVA（Grouped Value Attention）**：`HV` 可大于 `H`（value head 数量多于 qk head 数量），通过 `repeat_interleave` 展开。
- **参数**：
  - `q/k`：shape `[B, T, H, K]`
  - `v`：shape `[B, T, HV, V]`，`HV` 须整除 `H`
  - `g`：per-dimension 对数空间遗忘门，shape `[B, T, HV, K]`
  - `beta`：标量加权系数，shape `[B, T, HV]`
  - `scale`：缩放因子，默认 `1/sqrt(K)`
  - `initial_state`：初始状态 `[B, HV, K, V]`
  - `output_final_state`：是否返回最终状态
- **返回**：`(o, S)`，`o` 形状 `[B, T, HV, V]`

**`naive_chunk_kda(q, k, v, g, beta, scale, initial_state, output_final_state, chunk_size)`**

- **原理**：将序列切成 chunk 后分块处理。在每个 chunk 内用矩阵形式求解内部注意力矩阵 `A`（通过前向代入法 solve lower-triangular），然后计算 `w = A @ (gated k)` 和 `u = A @ v` 作为对状态的修正；跨 chunk 的状态更新采用类似递归的方式。
- **参数**：同上，另有 `chunk_size`（默认 64）
- **返回**：`(o, S)`

---

### 3. `gate.py`（KDA 遗忘门）

**职责**：计算 KDA 遗忘门（forget gate）。支持两种模式：标准模式（`-exp(A_log) * softplus(g + dt_bias)`）和有下界模式（`lower_bound * sigmoid(exp(A_log) * g)`）。提供前向、反向及融合了 chunk cumsum 的版本。

#### Triton 核函数

| 核函数 | 说明 |
|---|---|
| `kda_gate_fwd_kernel` | 逐 token 计算门值（可选 dt_bias、beta sigmoid、lower_bound） |
| `kda_gate_bwd_kernel` | 计算门的梯度 `dg`、`dA_log`、`dbeta` |
| `kda_gate_chunk_cumsum_vector_kernel` | 门计算 + chunk 内前缀累积和（一次 pass 完成） |

#### Python 函数

| 函数 | 说明 |
|---|---|
| `naive_kda_gate(g, A_log, dt_bias, output_dtype)` | PyTorch 参考实现，计算 `-exp(A_log) * softplus(g + dt_bias)` |
| `naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound, output_dtype)` | PyTorch 参考实现（下界模式） |
| `kda_gate_fwd(g, A_log, dt_bias, lower_bound, output_dtype)` | 调用 Triton 前向核，返回门值 |
| `kda_gate_bwd(g, A_log, dt_bias, dyg, lower_bound)` | 调用 Triton 反向核，返回 `(dg, dA_log, dbias)` |
| `fused_kda_gate(g, A_log, dt_bias, lower_bound, output_dtype)` | 带 autograd 支持的融合门计算（`KDAGateFunction.apply`） |
| `kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, ...)` | 门 + chunk cumsum，用于 chunk 前向的预处理步骤 |

#### 类

| 类 | 说明 |
|---|---|
| `KDAGateFunction` | `torch.autograd.Function`，封装 `kda_gate_fwd` 和 `kda_gate_bwd` |

---

### 4. `wy_fast.py`（WY 表示快速计算）

**职责**：利用 chunk 内已有的注意力矩阵 `A`，快速计算经过门调制的键 `w`（用于状态更新）和调制值 `u`（替换原始 v），以及可选的 `qg`（门调制 q）和 `kg`（跨 chunk 归一化的 k）。同时提供反向传播版本。

#### Triton 核函数

| 核函数 | 说明 |
|---|---|
| `recompute_w_u_fwd_kda_kernel` | 根据 A、k、v、beta、gk 计算 `w = A @ (beta*k*exp(gk))` 和 `u = A @ (beta*v)` |
| `prepare_wy_repr_bwd_kda_kernel` | 反向传播，从 `dw`、`du` 计算 `dk`、`dv`、`db`、`dg`、`dA` |

#### Python 函数

| 函数 | 说明 |
|---|---|
| `recompute_w_u_fwd(k, v, beta, A, gk, q, cu_seqlens, chunk_indices)` | 前向：输出 `(w, u, qg, kg)`；当 `disable_recompute=False` 时也在反向时调用以节省显存 |
| `prepare_wy_repr_bwd(k, v, beta, gk, A, dk, dw, du, dg, cu_seqlens, chunk_indices)` | 反向：输出 `(dk, dv, db, dg, dA)` |

---

### 5. `chunk_intra_token_parallel.py`（Token 并行 intra-chunk）

**职责**：intra-chunk 注意力的 token 并行实现。每个 token 独立获得一个 thread block，减少填充浪费，尤其适合变长序列。

#### Triton 核函数

| 核函数 | 说明 |
|---|---|
| `chunk_kda_fwd_kernel_intra_token_parallel` | 对每个 token `i`，计算它与当前 chunk 内所有 `j <= i` 的 `Aqk[i,j]`（q-k 注意力）和 `Akk[i,j]`（k-k 注意力） |

#### Python 函数

| 函数 | 说明 |
|---|---|
| `chunk_kda_fwd_intra_token_parallel(q, k, gk, beta, Aqk, Akk, scale, cu_seqlens, chunk_size, sub_chunk_size)` | 启动核函数，**原地写入** `Aqk` 和 `Akk` |

**注意**：`Akk` 的形状为 `[B, T, HV, BC]`（其中 `BC = sub_chunk_size`），仅存储每个 sub-chunk 的对角块；`Aqk` 形状为 `[B, T, HV, BT]`。

---

### 6. `chunk_intra.py`（Intra-chunk 完整实现）

**职责**：intra-chunk 的完整前向和反向流程。前向将 token 并行内核与跨 sub-chunk 的 off-diagonal 块计算及三角求解融合在一起；反向处理 `dAqk` 和 `dAkk` 的回传。

#### Triton 核函数

| 核函数 | 说明 |
|---|---|
| `chunk_kda_fwd_kernel_inter_solve_fused` | 融合核：计算 off-diagonal `Akk` 块 + 执行前向代入（lower-triangular solve），同时计算 off-diagonal `Aqk` 块 |
| `chunk_kda_bwd_kernel_intra_dqkb` | 反向：从 `dAqk`、`dAkk` 计算 `dq`、`dk`、`dbeta`、`dg` |

#### Python 函数

| 函数 | 说明 |
|---|---|
| `chunk_kda_fwd_intra(q, k, v, gk, beta, scale, cu_seqlens, chunk_size, chunk_indices, safe_gate, disable_recompute)` | 前向完整流程：先调用 token 并行算对角块，再调用融合核算 off-diagonal 并 solve；返回 `(w, u, qg, kg, Aqk, Akk)` |
| `chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_size, chunk_indices, safe_gate)` | 反向流程：从 `dAqk`/`dAkk` 出发，累加 `dq`、`dk`、`db`、`dg` |

---

### 7. `chunk_fwd.py`（分块前向编排）

**职责**：协调分块前向的各阶段，调用各子模块，最终输出 `o`（注意力输出）和 `final_state`（最终状态）。

#### Python 函数

**`chunk_kda_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, ...)`**

执行步骤（顺序）：
1. **门计算**：如果 `use_gate_in_kernel=True`，调用 `kda_gate_chunk_cumsum` 融合计算门并做 chunk cumsum；否则仅做 `chunk_local_cumsum`。
2. **Intra-chunk**：调用 `chunk_kda_fwd_intra`，得到 `w`、`u`、`qg`、`kg`、`Aqk`、`Akk`。
3. **CP 预处理**（可选）：分布式 Context Parallel 时，调用 `chunk_gated_delta_rule_fwd_h_pre_process` 处理初始状态。
4. **状态更新（Inter-chunk）**：调用 `chunk_gated_delta_rule_fwd_h`，从 `kg`、`w`、`u`、`g` 递推所有 chunk 的隐状态 `h` 及修正值 `v_new`。
5. **CP 压缩**（可选）：调用 `compress_h0` 压缩初始状态。
6. **输出计算**：调用 `chunk_gla_fwd_o_gk`，用 `q`、`v_new`、`Aqk`、`h` 计算最终输出 `o`。
7. **内存优化**：根据 `disable_recompute` 标志决定是否释放中间张量。

返回：`(o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)`

---

### 8. `chunk_bwd.py`（分块反向编排）

**职责**：完整的分块 KDA 反向传播，计算所有输入的梯度。包含两个 Triton 核及一个编排函数。

#### Triton 核函数

| 核函数 | 说明 |
|---|---|
| `chunk_kda_bwd_kernel_dAv` | 计算 `dA`（注意力矩阵梯度）和 `dv`（值梯度），通过 `do @ v^T` |
| `chunk_kda_bwd_kernel_wy_dqkg_fused` | 融合核：从 `dh` 出发，同时计算 `dq`、`dk`、`dg`、`db`、`dA`（与 WY 表示有关的梯度） |

#### Python 函数

| 函数 | 说明 |
|---|---|
| `chunk_kda_bwd_dAv(q, k, v, do, A, scale, cu_seqlens, chunk_size, chunk_indices)` | 封装 `chunk_kda_bwd_kernel_dAv` |
| `chunk_kda_bwd_wy_dqkg_fused(q, k, v, v_new, g, beta, A, h, do, dh, dv, scale, ...)` | 封装 `chunk_kda_bwd_kernel_wy_dqkg_fused` |
| `chunk_kda_bwd(q, k, v, beta, Aqk, Akk, scale, initial_state, do, dht, g, ...)` | 完整反向编排：① 重计算 w/u（若未缓存）→ ② `chunk_kda_bwd_dAv` → ③ `chunk_gated_delta_rule_bwd_dhu` → ④ `chunk_kda_bwd_wy_dqkg_fused` → ⑤ `chunk_kda_bwd_intra` → ⑥ 反向 cumsum & 门反向；返回 `(dq, dk, dv, db, dg, dh0, dA, dbias)` |

---

### 9. `chunk.py`（对外主 API）

**职责**：将前向/反向包装为 `torch.autograd.Function`，并提供完整参数校验、GVA 支持、CP 支持等。

#### 类

**`ChunkKDAFunction(torch.autograd.Function)`**

- `forward`：调用 `chunk_kda_fwd`，保存反向所需张量至 `ctx`
- `backward`：调用 `chunk_kda_bwd`，处理 L2norm 梯度（若启用）

#### 主 API 函数

**`chunk_kda(q, k, v, g, beta, scale, initial_state, output_final_state, ...)`**

完整参数说明：

| 参数 | 说明 |
|---|---|
| `q, k` | shape `[B, T, H, K]`，qk 注意力张量 |
| `v` | shape `[B, T, HV, V]`，`HV >= H`（GVA 时 `HV > H`） |
| `g` | 对数空间遗忘门，shape `[B, T, HV, K]` |
| `beta` | 更新率，shape `[B, T, HV]` |
| `scale` | 缩放，默认 `1/sqrt(K)` |
| `initial_state` | 初始隐状态 `[N, HV, K, V]`，须为 `float32` |
| `use_qk_l2norm_in_kernel` | 是否在核内部对 q/k 做 L2 归一化 |
| `use_gate_in_kernel` | 是否在核内部融合门计算（需同时传 `A_log`） |
| `A_log` | 门参数，shape `[HV]`（`use_gate_in_kernel=True` 时需要） |
| `dt_bias` | 时间步偏置，shape `[HV*K]` |
| `cu_seqlens` | 变长序列累积长度，shape `[N+1]` |
| `safe_gate` | 是否使用 lower_bound sigmoid 门（需配合 `lower_bound`） |
| `lower_bound` | 遗忘门下界（推荐 `-5`），配合 `safe_gate=True` |
| `disable_recompute` | 禁用梯度重计算（省时但耗显存） |
| `return_intermediate_states` | 推理时返回所有 chunk 的中间状态 h |
| `cp_context` | Context Parallel 上下文（多机分布式） |
| `transpose_state_layout` | 是否使用 `[V, K]` 转置状态布局 |

---

### 10. `fused_recurrent.py`（融合递归实现）

**职责**：逐 token 顺序递归实现，用单个 Triton 核覆盖整个前向流程。适合推理（短序列、decode 阶段）或作为 chunk 实现的基准对比。

#### Triton 核函数

**`fused_recurrent_kda_fwd_kernel`**

每个线程块负责一个 `(batch, value_head, K_block, V_block)` 的组合，在 token 维度顺序迭代：
- 加载 q/k/v/g/beta
- 更新隐状态：`h *= exp(gk)`，`h += beta * k ⊗ (v - k^T h)`
- 计算输出：`o = q^T h`
- 支持连续批处理（`ssm_state_indices`）、推测解码（`num_accepted_tokens`）、变长序列

#### Python 函数

| 函数 | 说明 |
|---|---|
| `fused_recurrent_kda_fwd(q, k, v, g, beta, ...)` | 启动 Triton 核，返回 `(out, final_state)` |
| `fused_recurrent_kda(q, k, v, g, beta, ...)` | 公开 API，含参数校验，调用 `fused_recurrent_kda_fwd` |

---

### 11. `backends/__init__.py`（后端注册表）

**职责**：创建 `kda_registry`，注册两个可选后端（FlashKDA 和 TileLang），通过 `dispatch` 机制在运行时自动选择最优实现。

```python
kda_registry = BackendRegistry("kda")
kda_registry.register(FlashKDABackend())    # 优先级 3（最高）
kda_registry.register(KDATileLangBackend()) # 次优先
```

---

### 12. `backends/flashkda.py`（FlashKDA 后端）

**职责**：封装 MoonshotAI 开源的 [FlashKDA](https://github.com/MoonshotAI/FlashKDA) CUTLASS 实现，仅用于推理（`torch.no_grad()` / `torch.inference_mode()`）。

#### 类 `FlashKDABackend`

| 方法 | 说明 |
|---|---|
| `chunk_kda_verifier(...)` | 检查条件是否满足（仅支持 bf16、K=V=128、非 GVA、推理模式、`safe_gate=True` 等）。返回 `(bool, str\|None)`——第一个值为是否可用，第二个值为不可用时的原因描述。 |
| `chunk_kda(...)` | 调用 `flash_kda.fwd(...)` 执行前向，返回 `(out, final_state)` 元组：`out` 形状 `[B, T, HV, V]`；`final_state` 形状 `[N, HV, K, V]` 若 `output_final_state=True` 否则为 `None`。若 `flash_kda` 包未安装则抛出 `ImportError`。 |

---

### 13. `backends/tilelang/`（TileLang 后端）

**职责**：使用 [TileLang](https://github.com/tile-ai/tilelang) DSL 编写的另一种 GPU 后端，提供 chunk 前向和反向实现（尤其是 dq/dk/dg 的计算）。

| 文件 | 说明 |
|---|---|
| `__init__.py` | 定义 `KDATileLangBackend` 类，注册 `chunk_kda` 和 `chunk_kda_bwd_wy_dqkg_fused` |
| `chunk_bwd_dqkg.py` | TileLang 实现的反向 dq/dk/dg 计算核 |

---

## 三、文件间依赖关系

```
外部公共 API 调用链：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

chunk_kda (chunk.py)
  ├── [前向] ChunkKDAFunction.forward
  │     ├── gate.py: kda_gate_chunk_cumsum     ← 门 + cumsum
  │     ├── chunk_fwd.py: chunk_kda_fwd
  │     │     ├── gate.py: kda_gate_chunk_cumsum / fla.ops.utils: chunk_local_cumsum
  │     │     ├── chunk_intra.py: chunk_kda_fwd_intra
  │     │     │     ├── chunk_intra_token_parallel.py: chunk_kda_fwd_intra_token_parallel
  │     │     │     └── wy_fast.py: recompute_w_u_fwd
  │     │     ├── fla.ops.common: chunk_gated_delta_rule_fwd_h   ← inter-chunk 状态更新
  │     │     └── fla.ops.gla: chunk_gla_fwd_o_gk               ← 最终输出 o
  │     └── fla.modules.l2norm: l2norm_fwd  (若 use_qk_l2norm_in_kernel)
  │
  └── [反向] ChunkKDAFunction.backward
        ├── gate.py: kda_gate_bwd / kda_gate_chunk_cumsum
        ├── chunk_bwd.py: chunk_kda_bwd
        │     ├── wy_fast.py: recompute_w_u_fwd           ← 重计算（若未缓存）
        │     ├── chunk_bwd.py: chunk_kda_bwd_dAv
        │     ├── fla.ops.common: chunk_gated_delta_rule_bwd_dhu  ← dh, dh0
        │     ├── chunk_bwd.py: chunk_kda_bwd_wy_dqkg_fused
        │     └── chunk_intra.py: chunk_kda_bwd_intra
        └── fla.modules.l2norm: l2norm_bwd  (若 use_qk_l2norm_in_kernel)


fused_recurrent_kda (fused_recurrent.py)
  └── fused_recurrent_kda_fwd            ← 单核顺序递归，无外部模块依赖


backends/ （运行时自动 dispatch）
  ├── flashkda.py: FlashKDABackend
  │     └── flash_kda（外部 CUTLASS 包）   ← 仅推理
  └── tilelang/: KDATileLangBackend
        └── TileLang DSL 核               ← 可选训练/推理后端
```

### 模块内部依赖矩阵

| 文件 | 依赖的内部模块 |
|---|---|
| `__init__.py` | `chunk.py`, `fused_recurrent.py` |
| `chunk.py` | `chunk_fwd.py`, `chunk_bwd.py`, `fla.modules.l2norm`, `fla.ops.backends`, `fla.ops.cp`, `fla.ops.utils` |
| `chunk_fwd.py` | `gate.py`, `chunk_intra.py`, `fla.ops.common`, `fla.ops.cp`, `fla.ops.gla`, `fla.ops.utils` |
| `chunk_bwd.py` | `gate.py`, `chunk_intra.py`, `wy_fast.py`, `fla.ops.common`, `fla.ops.cp`, `fla.ops.utils` |
| `chunk_intra.py` | `chunk_intra_token_parallel.py`, `wy_fast.py`, `fla.ops.utils` |
| `chunk_intra_token_parallel.py` | `fla.ops.utils.op`, `fla.utils` |
| `gate.py` | `fla.ops.utils.index`, `fla.ops.utils.op`, `fla.ops.utils.softplus`, `fla.utils` |
| `wy_fast.py` | `fla.ops.utils`, `fla.ops.utils.op`, `fla.utils` |
| `fused_recurrent.py` | `fla.ops.utils.op`, `fla.ops.utils.softplus`, `fla.utils` |
| `naive.py` | `torch`, `einops` |
| `backends/__init__.py` | `backends/flashkda.py`, `backends/tilelang/` |
| `backends/flashkda.py` | `fla.ops.backends`（基类），`flash_kda`（外部包） |
| `backends/tilelang/__init__.py` | `backends/tilelang/chunk_bwd_dqkg.py` |

---

## 四、核心算法背景

**KDA（Key-Decay Attention）** 是一种带有遗忘机制的线性注意力变体，其递推形式为：

```
S_t = diag(exp(g_t)) * S_{t-1} + beta_t * k_t ⊗ (v_t - k_t^T S_{t-1})
o_t = scale * q_t^T S_t
```

其中：
- `S` 是 `[K, V]` 的外积状态矩阵
- `g_t ∈ R^K` 是每维独立的对数空间遗忘门（per-dimension decay）
- `beta_t` 是更新率，用于控制写入强度
- 遵循 delta-rule：仅写入残差 `v_t - k_t^T S_{t-1}`，避免重复累积已存储的信息

**分块算法（Chunk Algorithm）**：

将序列分为大小为 `BT=64` 的 chunk，每个 chunk 内先求解注意力矩阵 `A`（通过正向代入求解下三角线性系统），得到修正键 `w` 和修正值 `u`，再在 chunk 间做递推更新状态。最终输出由当前状态贡献（inter-chunk）和 chunk 内注意力（intra-chunk）两部分相加得到。

此设计使得：
1. **intra-chunk** 计算可以高度并行（矩阵乘法）
2. **inter-chunk** 状态更新退化为标准 gated delta-rule 递推
3. 整体复杂度 `O(T * K * V)` 而非 softmax attention 的 `O(T^2 * V)`

---

## 五、快速查找索引

| 我想了解... | 看这个文件 |
|---|---|
| 算法最简单的实现 | `naive.py` |
| 主训练/推理入口 | `chunk.py: chunk_kda` |
| 推理单步 decode | `fused_recurrent.py: fused_recurrent_kda` |
| 遗忘门的计算方式 | `gate.py` |
| intra-chunk 注意力矩阵 | `chunk_intra.py`, `chunk_intra_token_parallel.py` |
| 状态更新的 w/u 向量 | `wy_fast.py` |
| 前向整体流程 | `chunk_fwd.py` |
| 反向整体流程 | `chunk_bwd.py` |
| FlashKDA 推理加速 | `backends/flashkda.py` |
| TileLang 替代后端 | `backends/tilelang/` |
