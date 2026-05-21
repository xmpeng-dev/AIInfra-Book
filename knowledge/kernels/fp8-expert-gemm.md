# FP8 Expert GEMM Optimization on AMD MI355X (gfx950)

> **目标：** 在 Primus/Megatron DSv3 训练中，用 FP8 Expert GEMM 替代 BF16 grouped GEMM，利用 MI355X 的 2x FP8 FLOPS 优势加速 MoE 层  
> **硬件：** AMD MI355X (gfx950), ROCm 7.1, 单节点 8 卡  
> **模型：** DeepSeek-V3, 256 experts, EP=8, 32 experts/GPU, 4 layers, MBS=2  
> **日期：** 2026-04-02

---

## 1. 背景与动机

### 1.1 MoE Expert GEMM 的核心计算

DSv3 的 MoE 层中，每个 GPU 需要计算 32 个 expert 的 fc1 和 fc2 GEMM：

```
Forward:  C[e] = A[tokens_e] @ W[e]      for e = 0..31
          A: [total_M, K=7168]  BF16
          W: [E=32, K=7168, N=2048]  BF16
          C: [total_M, N=2048]  BF16
```

Backward 包括 dA (activation gradient) 和 dW (weight gradient)，共 3 组 grouped GEMM。

### 1.2 FP8 的理论优势

MI355X (gfx950) 原生支持 OCP E4M3FN (`float8_e4m3fn`)，理论上 FP8 矩阵乘算力是 BF16 的 2 倍。目标是将此优势转化为 Expert GEMM 的实际加速。

### 1.3 端到端性能基线

所有测试使用相同的 FP8 pretrain config (`deepseek_v3-FP8-pretrain.yaml`)，仅切换 Expert GEMM 路径：

| 配置 | 迭代时间 | TFLOP/s/GPU | tokens/s/GPU |
|------|---------|------------|-------------|
| BF16 Legacy GG (`LEGACY_GG=True`) | **1980 ms** | 714 | 33,100 |
| FP8 _scaled_mm loop | 2260 ms | 624 | 29,000 |
| FP8 Fused Triton (优化后) | 2290 ms | 617 | 28,600 |

FP8 Expert GEMM 路径比 BF16 baseline 慢约 **14%** (~280 ms/iter)。

---

## 2. 技术方案探索

### 2.1 方案一：hipBLASLt GroupedGemm API (FP8)

**思路：** 直接调用 `hipblaslt_ext::GroupedGemm` 实现单次 batched FP8 GEMM。

**结果：** ROCm 7.1 的 `hipblaslt_ext::GroupedGemm` **不支持 FP8** 数据类型。通过系统测试所有 (opA, opB, dtype) 组合确认，只有 BF16 grouped GEMM 可用。

**验证代码：** `test_hipblaslt_fp8_ops.py` — 枚举测试 hipBLASLt grouped GEMM 的 FP8 支持。

### 2.2 方案二：torch._scaled_grouped_mm

**思路：** 使用 PyTorch 内置的 `torch._scaled_grouped_mm` API。

**结果：** 发现 Bug — 内部验证层要求 `float8_e4m3fnuz` (AMD FNUZ)，但底层 CK kernel 要求 `float8_e4m3fn` (OCP)，两者矛盾导致 API 不可用。

### 2.3 方案三：C++ _scaled_mm 循环

**思路：** 将 Python 层的 `for e in range(E): torch._scaled_mm(...)` 循环移到 C++ extension，减少 Python dispatch 开销。

**实现：**
```cpp
// hipblaslt_fp8_grouped_gemm.cpp
for (int64_t e = 0; e < E; e++) {
    auto A_e = A_fp8.narrow(0, offset, m);
    auto W_e_t = W_fp8_NK[e].t();  // column-major
    auto C_e = at::_scaled_mm(A_e, W_e_t, scale_a, scale_b,
                               /*bias=*/c10::nullopt, /*scale_result=*/c10::nullopt,
                               /*out_dtype=*/torch::kBFloat16, /*use_fast_accum=*/false);
    chunks.push_back(C_e);
    offset += m;
}
return torch::cat(chunks, 0);
```

**结果：** C++ loop vs Python loop 性能相当 — 瓶颈是 hipBLASLt kernel 本身的 dispatch，而非 Python 开销。

### 2.4 方案四：FP8 Fused Triton Grouped GEMM（最终方案）

**思路：** 复用 BF16 fused Triton grouped GEMM 的 2D grid 模式（`bf16_fused_grouped_gemm.py`），改造为 FP8 版本。单次 kernel launch 处理所有 expert。

**核心设计：**

```
Grid: (E, max_tiles_per_expert)

每个 Triton program:
  1. 从 expert_starts[eid] 推算自己负责的 (expert, m_tile, n_tile)
  2. 加载 FP8 的 A tile 和 B tile
  3. tl.dot(a.to(float8e4nv), b.to(float8e4nv), acc, out_dtype=float32)
  4. Dequant: acc *= scale_a * scale_b
  5. 写出 BF16 结果
```

**优势：**
- 单次 kernel launch（消除 32 次 _scaled_mm dispatch）
- 无需 torch.cat（输出直接写入连续内存）
- 无需 CPU tile map（2D grid 自动映射）
- FP8 tensor core 加速

---

## 3. 实现细节

### 3.1 新增核心代码

**文件：** `primus/backends/megatron/core/fusions/fp8_grouped_gemm.py`

新增函数：
- `_fp8_fused_fwd_kernel`: Triton JIT kernel，2D grid `(E, max_tiles)`
- `_fp8_fused_grouped_fwd`: Forward Python 入口
- `_fp8_fused_grouped_dA`: Backward dA Python 入口（接收缓存的 W_fp8_NK）
- `_fp8_fused_bwd`: 统一 backward dispatcher（fused Triton dA + BF16 dW）
- `_use_fused_fp8_triton()`: 环境变量控制开关 (`PRIMUS_FP8_FUSED_TRITON`)

### 3.2 Dispatch 优先级

```python
def fp8_grouped_gemm_forward_prequantized_w(...):
    if _use_fused_fp8_triton():        # 1st: Fused Triton (new, default)
        A_fp8, A_si = _fast_per_tensor_quant(A)
        return _fp8_fused_grouped_fwd(A_fp8, A_si, B_fp8, B_si, tpe)

    if _use_hipblaslt_fp8():           # 2nd: _scaled_mm loop
        ...

    # 3rd: Old Triton 1D grid (fallback)
    ...
```

### 3.3 Weight Cache 更新

**文件：** `primus/backends/megatron/core/extensions/primus_turbo.py`

`_refresh_fp8_weight_cache` 更新为同时缓存 KN 和 NK layout（fused Triton backward dA 需要 NK layout），使用 `_per_tensor_quant_W` 量化（无需 `_prequant_W_for_scaled_mm` 的双重量化）。

### 3.4 Autograd 集成

`Fp8GroupedGemmPrequantWFunction`:
- Forward: 保存 `W_fp8_NK`, `W_fp8_KN`, `B_si_scalar` 用于 backward
- Backward: `_fp8_fused_bwd` → fused Triton dA + BF16 `_bfloat16_grouped_gemm_weight_grad` dW

---

## 4. 性能分析

### 4.1 微 Benchmark（单次 GEMM 调用，E=32, K=7168, N=2048, M=8192）

| 操作 | _scaled_mm loop | Fused Triton | 加速比 |
|------|----------------|-------------|--------|
| FP8 FWD GEMM | 0.562 ms | 0.310 ms | **1.81x** |
| FP8 dA GEMM | 0.514 ms | 0.372 ms | **1.38x** |
| Full FWD (quant+GEMM) | 0.755 ms | 0.524 ms | **1.44x** |

与 BF16 baseline 对比：

| 操作 | BF16 gg_ops.gmm | FP8 Fused Triton | 加速比 |
|------|-----------------|-----------------|--------|
| FWD | 0.983 ms | 0.524 ms | **1.88x** |

### 4.2 Autograd 全路径（FWD + BWD）

| 路径 | 耗时 | 差异 |
|------|------|------|
| FP8 Fused Triton | 1.798 ms | +0.186 ms |
| BF16 gg_ops.gmm | 1.612 ms | baseline |

Per iteration (16 GEMM calls): **仅 +3 ms**

### 4.3 端到端 DSv3 训练

| 配置 | 迭代时间 | 差异 |
|------|---------|------|
| BF16 Legacy GG | 1980 ms | baseline |
| FP8 Fused Triton | 2290 ms | +310 ms |

**关键发现：** Expert GEMM 仅贡献 +3 ms 差异，剩余 ~277 ms 来自 `PrimusFP8GroupedMLP` vs `GroupedMLP` 模块框架级开销。

### 4.4 Overhead Breakdown

```
端到端差距:                        +280 ms
├── Expert GEMM (16 calls):         +3 ms    (1%)
└── 模块框架开销:                   +277 ms  (99%)
    ├── PrimusFP8GroupedMLP 模块 overhead
    ├── Weight cache 管理
    ├── Autograd Function 7-tensor apply()
    ├── tokens_per_expert D2H sync
    └── 额外内存压力 (55.6% vs 49.9%)
```

---

## 5. 辅助发现

### 5.1 use_fast_accum

`torch._scaled_mm(..., use_fast_accum=True)` 在 MI355X 上仅有 ~2% 差异，不值得追求。

### 5.2 FP8 数据类型

MI355X (gfx950) 应使用：
- **Triton**: `tl.float8e4nv` (OCP E4M3FN, 原生支持，无 upcast)
- **PyTorch**: `torch.float8_e4m3fn`
- **最大值**: 448.0
- **避免**: `tl.float8e4b8` (AMD FNUZ, gfx942 专用，在 gfx950 上会 upcast 到 fp16)

### 5.3 hipBLASLt FP8 限制

- `hipblaslt_ext::GroupedGemm` 不支持 FP8（ROCm 7.1）
- `torch._scaled_grouped_mm` 有 FNUZ/E4M3FN 冲突 Bug
- `torch._scaled_mm` 单次调用可用，但需要 per-expert 循环

---

## 6. 下一步优化方向

Expert GEMM kernel 已接近最优（FP8 组件级比 BF16 快 9.6%）。真正的瓶颈在模块框架层。

### 6.1 减少 PrimusFP8GroupedMLP 框架开销

- **简化 Autograd Function**: 减少 `apply()` 的输入参数数量（当前 7 个 tensor）
- **消除 D2H sync**: `tokens_per_expert.tolist()` 触发 GPU→CPU 同步，改为全 GPU 操作
- **轻量化 weight cache**: 当前每次 `_refresh_fp8_weight_cache` 检查 step 号，可用更轻的标志位

### 6.2 模块级融合

- 将 `PrimusFP8GroupedMLP` 的 forward 逻辑向 `GroupedMLP` 的简洁模式靠拢
- 考虑将 FP8 Expert GEMM 集成到 Megatron 的 `GroupedMLP` 中，复用其高效框架

### 6.3 Triton Kernel 调优

- dA 路径（N_OUT=7168）的 BLOCK_N=256 产生 28 tiles，可调优为更小 block
- 探索 persistent kernel 模式减少 grid launch 开销
- 量化融合：将 `_fast_per_tensor_quant` 的 amax+scale+cast 融入 GEMM prologue

### 6.4 ROCm/PyTorch 升级

- 跟踪 `hipblaslt_ext::GroupedGemm` FP8 支持（未来 ROCm 版本）
- 跟踪 `torch._scaled_grouped_mm` Bug 修复
- 一旦可用，单次 API 调用可消除所有 dispatch 开销

---

## 7. 相关代码文件

| 文件 | 说明 |
|------|------|
| `primus/backends/megatron/core/fusions/fp8_grouped_gemm.py` | FP8 grouped GEMM 主实现（Triton kernel + autograd） |
| `primus/backends/megatron/core/fusions/bf16_fused_grouped_gemm.py` | BF16 fused Triton grouped GEMM（参考实现） |
| `primus/backends/megatron/core/fusions/csrc/hipblaslt_fp8_grouped_gemm.cpp` | C++ _scaled_mm loop extension |
| `primus/backends/megatron/core/extensions/primus_turbo.py` | PrimusFP8GroupedMLP 模块 + weight cache |
| `primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py` | 模块选择逻辑 |

---

## 8. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `PRIMUS_FP8_FUSED_TRITON` | `1` | 启用 FP8 Fused Triton Grouped GEMM |
| `PRIMUS_FP8_USE_HIPBLASLT` | `1` | 启用 _scaled_mm loop 路径 (fused Triton 优先) |
| `TRITON_FP8_EXPERT_GEMM` | `True` | 在 run.sh 中启用 FP8 Expert GEMM |
| `LEGACY_GG` | `False` | 使用 PrimusFP8GroupedMLP (False) 或 GroupedMLP (True) |
