# hipBLAS

> **Repo:** `ROCm/hipBLAS` &nbsp; **Local path:** `3rd/hipBLAS/`
> **Snapshot:** `23b26a0` &nbsp; `2025-09-24` &nbsp; on branch `develop_deprecated`
> **Size:** ~25 MB &nbsp; **License:** MIT
> **Distilled:** 2026-05-21
> **Status:** retired —— 仓库已搬入 [`ROCm/rocm-libraries`](https://github.com/ROCm/rocm-libraries) monorepo,本文作为 marshalling-layer pattern 的历史参考留存。

## TL;DR

hipBLAS 的全部价值是一个设计模式:**一份公共 BLAS API,两个可互换的后端实现**(AMD 走 rocBLAS,NVIDIA 走 cuBLAS)。README 原文一句话定性:"BLAS **marshalling library with multiple supported backends**"。它**不是**性能库 —— 完全不写 kernel,所有性能交给后端。今天 transformer 训练里实际被消费的是它的 sibling [`hipBLASLt`](#),hipBLAS 本身已经退役。但 marshalling pattern 本身仍然是我们设计 ROCm ↔ CUDA 可移植层(Megatron-Core 跨栈 shim)的最干净参考。

## 1. 库定位 (positioning)

- **一句话:** cuBLAS-v2 风格的 BLAS API 可移植层,把"调用方代码不改"作为唯一卖点,后端在 rocBLAS 或 cuBLAS 之间切换。
- **它是什么:**
  - 一份公共 C/C++ 头 `library/include/hipblas.h`(~25 800 行,doxygen 注释为主)。
  - 两份后端 marshalling 实现,每份就一个 `.cpp` 文件 —— `amd_detail/hipblas.cpp`(rocBLAS)与 `nvidia_detail/hipblas.cpp`(cuBLAS)。
  - 配套 `hipblas-test` / `hipblas-bench` / samples / Sphinx 文档。
- **它不是什么(非目标比功能列表更重要):**
  - 不写任何 kernel,不做调度,不做 fusion;性能问题一概在后端追。
  - 不暴露 FP8 GEMM、bias+activation fusion、grouped GEMM 这些现代 LLM 需求 —— 这些归 hipBLASLt。
  - 不再演进 —— 顶部 README CAUTION 明确:"The hipBLAS repository is retired, please use the ROCm/rocm-libraries repository."
- **谁在用:** 历史上是 ROCm 上 BLAS 应用 / CUDA 旧代码迁移的默认 wrapper;现代训练栈(`primus-turbo`、`CK`、`AITER`)在 GEMM 这条路径上一律绕过 hipBLAS,直接吃 hipBLASLt 或 rocBLAS 自身。

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| `library/include/hipblas.h` | 唯一公共 header,API + 类型 + doxygen 全部在这里 | 单文件 ~25 800 行 |
| `library/src/amd_detail/hipblas.cpp` | rocBLAS 后端实现(marshalling) | 整个后端就这一个 `.cpp` |
| `library/src/nvidia_detail/hipblas.cpp` | cuBLAS 后端实现(marshalling) | 整个后端就这一个 `.cpp` |
| `library/src/include/` | C++ exception → `hipblasStatus_t` 转换的内部头 | 防止异常逃出 C API |
| `clients/` | `hipblas-test` + `hipblas-bench` + `samples` | API 行为对照测试 |
| `docs/` | Sphinx + Doxygen 文档源 | conceptual / how-to / reference 三组 |

整个仓库的"实质代码"就是 **1 个 public header + 2 个 backend `.cpp`**。其余都是测试、文档、构建脚本。

## 3. 核心设计理念 (core design ideas)

### 3.1 Marshalling-layer pattern (整个库的全部内容)

README 原文逐字:"a Basic Linear Algebra Subprograms (BLAS) **marshalling library with multiple supported backends**. It sits between your application and a 'worker' BLAS library, where it marshals inputs to the backend library and marshals results to your application. **hipBLAS exports an interface that doesn't require the client to change, regardless of the chosen backend.**"

落到代码上的形态非常干净:一份 `hipblas.h`,两份 `.cpp` 实现同一份符号集合,build 系统按 `HIP_PLATFORM=amd|nvidia` 选其一进 final `.so`。`hipblasHandle_t` / `hipblasStatus_t` / `hipblasOperation_t` 等 enum / handle / status 都是 1:1 包装到下层 `rocblas_*` 或 `cublas*` 同义类型,绝大多数函数体就是三件事:参数翻译 → 调一次下层 → 错误码翻译。这 99% 的代码量就是整个库的"实现"。

### 3.2 Explicit non-goal: 不是性能库

hipBLAS 故意把"做什么"和"不做什么"在 README 第一段就钉死 —— 性能完全推给后端,自己只承诺 ABI 稳定 + 行为对齐 cuBLAS-v2。这个非目标的设计后果是双向的:好处是 public API 不会随每代新硬件指令震荡,任何一个 cuBLAS 旧应用拿过来改个前缀就能编;代价是无法支持任何"后端独有"的高级特性 —— FP8 input 的 scale tensor、bias + GeLU fusion、grouped GEMM 的 `tokens_per_expert` 偏移,这些都不在 hipBLAS 表面上,必须直接调 hipBLASLt 或 rocBLAS。把这一条写在 README 第一段而不是埋在 FAQ 里,是这个库做得最对的一件事。

### 3.3 hipBLAS vs hipBLASLt — API 拆成"稳定全集"和"轻量 fused"

ROCm 把 BLAS API 主动拆成了两层,每层目标互斥:

- **hipBLAS** —— 经典 BLAS 全集 + BLAS-Ex,严格对齐 cuBLAS-v2,语义稳定但表面窄,无 FP8、无 fusion。面向"老 cuBLAS 应用 1:1 迁移"。
- **hipBLASLt** —— 轻量 GEMM-only API,围绕 `MatmulDescriptor` / `MatmulPreference`,显式暴露 FP8 input、bf16/fp32 accumulate、bias + activation fusion、grouped GEMM。NVIDIA 侧对应 cuBLASLt。面向 transformer 训练。

这是一个有意识的"经典 wrapper / 现代 fused-op wrapper"分家。现代 LLM 训练栈在 GEMM 这条路径上消费的几乎都是 hipBLASLt 而不是 hipBLAS —— `CK` 的 grouped GEMM 后端、`primus-turbo` 的 FP8 expert GEMM、`AITER` 的 fused-MoE,都走 hipBLASLt。读这个仓库时如果发现 "FP8 / fusion 在哪里",答案是:不在这里,在 sibling 库里。

### 3.4 Deprecation 自身就是一条设计教训

把 BLAS 这种"随每代新硬件演进"的库放在自己独立 repo 维护,会随每代新指令(MFMA、MX 块缩放、scale tensor)反复破坏 release cycle —— rocBLAS / hipBLAS / hipBLASLt 之间的 ABI 必须互相对齐。ROCm 6.x 之后把所有 math libs(rocBLAS / hipBLAS / hipBLASLt / rocSOLVER / rocSPARSE / ...)收进 `ROCm/rocm-libraries` monorepo,一次性同步版本与 release,正是为了消除这个跨 repo 漂移。"独立 repo + 持续演进"在中等规模库上的失败模式,本仓库的退役轨迹是一份很直接的样本。

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| **Marshalling-layer pattern**:一份 public header + 每后端一个 `.cpp` 实现文件,build 时按平台挑一个;enum / handle / status 一一对应到 vendor 同义类型 | "上层调用代码不想为 AMD / NVIDIA 写两份"的诉求,同时让两个后端各自自由演进 | Megatron-Core 长期"原生支持 ROCm + CUDA 双栈"的方向 —— 通信原语(RCCL / NCCL)、CUB-style 工具函数、profiler / event API,都可以照本仓库模板做一层 `xpu_*` 头 + 两个 backend `.cpp`,call-site 只 include 一份,消除散落在 `megatron/core/` 各处的 `#ifdef HIP_PLATFORM_*` 分支 | 只解决 API 兼容,不解决性能差异;必须把"本层不承诺两边等价性能"写进 doc 第一句,避免下游误以为切后端不需要重新调参 |
| **Public-API freeze + 平行 backend dirs in one repo**(`amd_detail/` 与 `nvidia_detail/` 同级) | 防止两个后端的 API 偷偷漂移 —— 任何 backend 要加新参数都必须先动 public header,自动逼着另一侧同步实现或显式拒绝 | primus-turbo 若将来要公开给 CUDA 用户用,首选"一份 public 头 + 两个 backend dir"的形态,而不是 fork 一个 CUDA 版本仓库各自演进 | repo size 与 CI 时间×2;只适合 API 演进慢、稳定优先的层(粘合层,而非 fused-op 本身) |
| **Explicit non-goal as design tool** —— README 第一段就钉"不写 kernel、性能在后端"| 抽象层会被持续要求加 feature 直到自己变重 | 我们自己的所有 portability shim 都该复用这招:在 doc 顶部声明 "不为 perf 服务,要 perf 调底层"。具体场景:Megatron-Core 通信 shim、profiler shim、xpu allocator shim | 团队会反复想加 perf hack;需要 review 时按 doc 硬挡 |
| **API 拆成 stable-wide 与 lightweight-fused**(hipBLAS vs hipBLASLt) | "经典 API 兼容性"与"前沿 fused-op 演进速度"天然冲突,塞同一层会两边都做不好 | 我们若把 MoE / FP8 算子对外暴露,可以同时给"稳定全集"和"轻量 fused-only"两层:前者面向兼容,后者(类比 hipBLASLt)面向 transformer 训练,显式吃 FP8 + bias/activation + grouped。CK / primus-turbo 实际接的就是这种 fused 层 | 两套 API 同时维护文档/测试负担×2;fused 层版本演进更快,必须配清晰的 deprecation 政策 |

## 5. 与生态的关系 (ecosystem position)

```
hipBLAS  →  rocBLAS (AMD)    |   cuBLAS (NVIDIA)
                │
sibling: hipBLASLt (lightweight GEMM, FP8 + bias/activation fusion + grouped GEMM)
                ↑ 真正被 CK / primus-turbo / AITER 在 production 消费
```

在 AMD 算子栈里 hipBLAS 现在只是**历史兼容层**:任何 cuBLAS 旧代码 1:1 翻译过来,call-site 不改即可在 ROCm 上跑。新代码,尤其是 transformer 训练里的 GEMM,统一走 sibling 库 hipBLASLt —— 它是 [CK](composable-kernel.md)、[primus-turbo](primus-turbo.md)、[AITER](aiter.md) 在 production 实际拉的那一层,负责 FP8 / fusion / grouped GEMM。后续蒸馏 `hipblaslt.md` 时会反向引用本文。

## 6. 进一步阅读 / TODO

- [`ROCm/rocm-libraries`](https://github.com/ROCm/rocm-libraries) —— **本仓库的活体替代**。ROCm 已经把 hipBLAS / rocBLAS / hipBLASLt / rocSOLVER / rocSPARSE 等 math libs 全部收进这个 monorepo,版本与 release 一次同步;后续所有 hipBLAS 变更只在这里发生。日常工作只读这一份就够,本仓库仅作 marshalling pattern 的最小参考。
- `3rd/hipBLAS/library/include/hipblas.h` —— 唯一公共 header,任何 API 兼容性问题以它为准。
- `3rd/hipBLAS/library/src/amd_detail/hipblas.cpp` —— marshalling 模板的最短示例:一个函数体 = 参数翻译 + 一次 `rocblas_*` 调用 + status 翻译;整个 AMD 后端就这一个文件。
- 下一篇要写的 [`hipblaslt.md`](hipblaslt.md) —— hipBLAS 的"现代化版本",承接所有 FP8 / fusion / grouped GEMM 需求,是 CK / primus-turbo / AITER 真正消费的那一层。
