# Composable Kernel (CK)

> **Repo:** `ROCm/composable_kernel` &nbsp; **Local path:** `3rd/composable_kernel/`
> **Snapshot:** `b7c2a933f` &nbsp; `2026-05-20` &nbsp; on branch `develop`
> **Size:** ~313 MB &nbsp; **License:** MIT
> **Distilled:** 2026-05-21

## TL;DR

CK 是 AMD 的通用 ML kernel 模板库 — 角色上等价于 "AMD 版 CUTLASS"。它把
"一份模板覆盖多 arch × 多 dtype × 多 layout" 这件事做到极致,靠的是两根明
确写在 README 里的支柱:**tile-based programming model** 和
**tensor coordinate transformation**。把 layout 和 padding/permute/im2col 当成
compile-time 可组合的 transform,这个抽象是我们做 MoE permute、grouped
GEMM 偏移、kernel 模板复用时最值得借鉴的一招。

## 1. 库定位 (positioning)

- **一句话:** GPU 通用算子模板库 + 大规模预编译 instance 库,服务 ML 框架
  的高性能算子需求(GEMM / Conv / FMHA / Norm / Reduce / Fused MoE),
  AMD 的 CUTLASS-equivalent。
- **它是什么:**
  - 一套 header-only 的 tile-level 模板原语(register / LDS / global 的
    tile 抽象),配一套 templated kernel 模板。
  - 一个把模板按 (dtype, layout, tile shape, pipeline, arch) 笛卡尔积
    AOT 实例化出的 .a/.so 二进制库 (`library/`)。
  - 一套稳定的 Client API:用户只声明问题形状,工厂返回该形状下的全部
    instance,运行期做 best-of-N。
- **它不是什么:**
  - 不是端到端推理引擎(不管 graph fusion / scheduler / KV cache)。
  - 不直接面向 PyTorch tensor(粘合层在 AITER / primus-turbo / hipBLASLt)。
  - 不解决跨节点 / 集合通信(RCCL / XGMI 不归它管)。
- **谁在用:** `AITER`(GEMM 后端)、`primus-turbo`(部分 fused-op)、
  `hipBLASLt`(grouped/extended GEMM 后端之一)、PyTorch Inductor
  (`ck4inductor` pip wheel)、TheRock 多 arch 打包(`rocm_ck/`)。

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| `include/ck/` | **Legacy** templates 实现(Layer 1–2 的老版本) | 维护态,仍是 `library/` 里多数 instance 的实现 |
| `include/ck_tile/` | **新** tile-based 编程模型(推荐新代码使用) | 与老 ck 独立,host glue 更薄,接口更轻 |
| `library/` | Layer 3:AOT 预编译 instance 库 | 218 个 op-config 目录,~1700 个 instance `.cpp` |
| `client_example/` | Layer 4:Client API 用法示范 | 新接入者从这里入门 |
| `example/ck_tile/` | ck_tile 的最小可读示例集 | ck_tile 学习的主要参考 |
| `tile_engine/` + `codegen/` | 从 op spec 批量生成 ck_tile / 老 ck instance 源码 | 内置 codegen 后端,带 `operation_support_matrix.md` |
| `dispatcher/` + `python/ck4inductor/` + `rocm_ck/` | 上层 binding 与多 arch 打包边界 | Inductor 后端 / TheRock kpack 都在这里 |

CK 自己在 README 给出一张 4 层架构图(`README.md:18–23`):**Templated
Tile Operators → Templated Kernel and Invoker → Instantiated Kernel and
Invoker → Client API**。下一节按这张图组织。

## 3. 核心设计理念 (core design ideas)

### 3.1 Tile-based programming model

README 原文:"CK uses two concepts to achieve performance portability
and code maintainability: **a tile-based programming model** + algorithm
complexity reduction via Tensor Coordinate Transformation"
(`README.md:10–14`)。

这条 idea 的本质是:kernel 的最小复用单元不是 thread,也不是 warp,而是
**tile** —— 一块在 register / LDS / global 之间被显式搬运、由一组 thread
协同处理的张量片段。CK 把 `tile_window` / `static_distributed_tensor` /
`load_tile` / `store_tile` / `shuffle_tile` 全部抽象到 `ck_tile/core/` 下,
新写 kernel 时只 `#include "ck_tile/core.hpp"` 就能拿到这套语义,不必再手
算 thread→element 索引。它选 tile 而不选更细的抽象,是因为 tile 既是寄存
器分配的天然单位,也是 LDS 双缓冲、async copy、software pipeline 的天然
单位 —— 调度策略可以在 tile 边界上换,不动算法本身。

### 3.2 Tensor coordinate transformation 作为 layout 抽象

CK 的第二根支柱(`include/ck_tile/README.md:5–6`):"tensor coordinate
transformation, this is the core concept of layout/index transform
abstraction in both compiler time and run time."

具体形式是把 ND tensor 的 indexing / padding / permute / merge / unmerge
/ embed / xor swizzle / 卷积 im2col 等全部表达成 **compile-time 可组合的
transform primitive**。一个 tensor descriptor = 一串 transform 叠加 +
最底层一段连续物理 buffer。这样做的代价是模板更深、编译更慢;换来的好处
是 README 那句口号 "algorithm complexity reduction" 的兑现 —— "卷积、
batched GEMM、permute、reshape、padding 各自一套 kernel" 折叠成 "同一个
GEMM kernel + 不同 tensor descriptor"。这是 CK 区别于"为每个 layout 各
写一份 kernel"路线最显著的设计选择。

### 3.3 四层架构:Templated Tile Operators → Kernel + Invoker → Instantiated → Client API

CK 把整个调用链显式切成四层(README 4 层图):

- **Layer 1 — Templated Tile Operators**:tile/warp/block 三级原语,
  `ck_tile/core/` + `ck_tile/ops/*/warp,block`。
- **Layer 2 — Templated Kernel and Invoker**:把 tile 原语拼成完整
  kernel 模板(`gemm_kernel.hpp` / `device_gemm_*.hpp`),配一个 host
  侧 Invoker 负责 launch。模板参数全部裸露在类型上。
- **Layer 3 — Instantiated Kernel and Invoker**:对每条 (dtype, layout,
  tile shape, pipeline scheduler, pipeline version, arch) 走一次完整
  instantiation,产出 `.cpp` 文件 → `.a/.so`。
- **Layer 4 — Client API**:把 Layer 3 的 instance 用 type-erased
  base-class 指针暴露给下游,下游只声明问题形状,工厂返回该形状下的全部
  instance,运行期 best-of-N。

这四层是 CK 全部 "怎么写 kernel / 怎么编译 / 怎么暴露给框架" 的回答。换
新 arch / 新指令时,改 Layer 1 的 warp dispatcher 表 + Layer 3 加一组
instance,Layer 2 / 4 不动;换前端框架(Inductor / AITER / primus-turbo)
时,只动 Layer 4。

### 3.4 `ck_tile` 是新代码的唯一推荐入口,老 `ck/` 处于迁移态

`include/ck_tile/README.md:8` 原文:"`ck_tile` is independently from
the old ck ... We will have a transition period to pull everything from
old ck into `ck_tile`, stay tuned." `ck_tile` 不是对老 `ck/` 的小重构,
而是 Layer 1–2 的重写:它把每个 component 收成一个总头(`ck_tile/core.hpp`、
`ck_tile/ops/gemm.hpp` 等),host glue 更薄,模板参数列表显著变短。

读这个库时的实操含义:学习路径直接走 `include/ck_tile/` 与
`example/ck_tile/`;`include/ck/` 只用来读懂 `library/` 里现存 instance
的实现,不用来作为新写代码的模板。

### 3.5 Per-(dtype, arch) instance 枚举 — 模板派遣的构建侧后果

把模板参数全部暴露在类型上(Layer 2)之后,要让下游运行期能选最优 tile,
就只能在构建期把 (dtype, layout, tile shape, pipeline, padding, arch)
的笛卡尔积**手工列举**成成百上千个 `.cpp` 文件 —— 每个 `.cpp` 是一次完
整的 template instantiation,CMake 把它们打成 per-op 静态/共享库。`library/`
下因此长出 **~1700 个 instance `.cpp` / 218 个 op-config 目录**。

这是 CK 整套设计的逻辑闭环,也是它与 Triton-only 路线最大的对比:**autotune
留在编译期 + 运行期实测线性扫,运行期完全不做 JIT、不做 template 实例化。**
代价具体而直接 —— 编译慢(`-DDTYPES=` 用来子集化以加速)、二进制体量大、
新 arch 上新指令需要重编整套。CK 接受这个代价,换来的是没有 JIT 抖动、
可预测的 runtime,以及"模板细节藏在 .so 后"的稳定 ABI。

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| **Tensor coordinate transform 作为 layout 抽象** —— 把 padding / permute / merge / unmerge / xor swizzle 等都做成 compile-time 可组合 transform,kernel 主体只认 tensor descriptor | "为每种 layout 重写一份 kernel" 的爆炸 | DSv3 MoE permute / unpermute(`primus/backends/megatron/core/fusions/`)、grouped GEMM 的 `tokens_per_expert` 偏移、FP8 expert GEMM 的 NK ↔ KN cache — 这些今天都是按 layout 各写一支 Triton kernel,统一到一组 descriptor 上能减一类回归 | 模板深度上来后编译时间会显著恶化,适合在稳定下来的 fused-op 上做,不适合每天都改的实验 kernel |
| **Tile / policy 作为一等公民类型**(Pipeline = `Problem` × `Policy` × `Scheduler`) | "搬数据策略"(LDS layout / async copy / SW pipeline 阶段数)与"算什么"(dtype + shape)耦在同一个 kernel 里,加 async direct-to-LDS 或 weight preshuffle 时必须复制整支 kernel | 我们自写的 HIP / Triton kernel 模板(MonolithMoE super-kernel、`fp8_grouped_gemm.py`、`bf16_fused_grouped_gemm.py`)可以把 tile shape 与 pipeline policy 拆成独立类型/`@triton.heuristics`,后续加新调度策略不动 kernel 主体 | Triton 没有 C++ 那么强的类型系统,policy 只能靠 `tl.constexpr` + 命名约定模拟,需要在 review 时人为守住边界 |
| **Instance enumeration + per-(dtype, arch) precompiled shards**(Layer 3) | Triton JIT 每条新 shape 都要重新编译,首调延迟和缓存抖动是生产瓶颈 | 当我们把 `PrimusFP8GroupedMLP` 这类 fused-op 沉淀成正式 op 库时,应当走 "编译期枚举 + 运行期 best-of-N + `op_ptr` cache" 而不是依赖 Triton JIT;同时这是未来落地 gfx950 新指令(`mfma_scale_f32_*_f8f6f4` / MX scale)最干净的扩展点 | 编译时间和二进制体量会立刻上一个量级,需要像 CK 的 `-DDTYPES=` 一样提供 dtype 子集化开关,并把这层 CI 与日常开发 CI 分开 |
| **Warp dispatcher 用 partial template specialization 表把 (dtype × wave-tile) 路由到 MFMA functor**(Layer 1 末端) | `#if defined(__gfx942__) ... #elif __gfx950__` 在 kernel 主体里发散 | 我们自己包 MFMA / WMMA wrapper(`primus-turbo` 的 AMD intrinsic 层、cco super-kernel 的 inner-loop)时,可以借这套 "(dtype, M, N, K, transpose) → functor" 的路由表形式,把 arch 分支收敛到一个文件 | 表项数会随支持组合线性增长,需要配 codegen 脚本(参考 `tile_engine/`)而不是手写 |
| **Client API 三件套**(`MakeArgumentPointer` + `IsSupportedArgument` + `Run`)+ `InstanceFactory<>::GetInstances()` | 下游既要 best-of-N 又要稳定 ABI,两者天然冲突 | 给 primus-turbo / AITER 风格的 op 设计 Python 入口时,可以照搬这套:Python 只声明问题形状,C++ 工厂返回 type-erased instance 向量,首次实测、之后按 `op_ptr` 指针 cache | "best-of-N" 的实测会污染第一次 step 的耗时,需要 warmup 与持久化 cache 协同 |
| **Dev iteration 5 s 快通道**(`script/cmake-ck-dev.sh --minimal`,关掉 instance / profiler / examples / tests,configure 从 ~150 s 降到 ~5 s) | 大型 header-only 模板库 reconfigure 慢到打断思路 | 我们任何 header-only 的算子库(将来 primus-turbo 子模块)都该提供同形的 `--minimal` preset,把 instance 编译与 header 改动解耦 | 只解决 configure,不解决单个 instance 编译本身的代价 |

## 5. 与生态的关系 (ecosystem position)

```
PyTorch / Inductor / framework
        │
        ├── primus-turbo  ─┐
        ├── AITER  ────────┼──► CK (Client API + instance factory)
        ├── hipBLASLt  ────┤            │
        └── ck4inductor  ──┘            ▼
                              ck_tile / ck templates
                                       │
                                       ▼
                              HIP / __builtin_amdgcn_mfma_*
                                       │
                              rocm_ck (TheRock 多 arch kpack)
```

CK 在 AMD 算子栈里处于"上承框架、下接 HIP intrinsic"的中枢位置:它给
`AITER` 提供 GEMM kernel 实现,被 `hipBLASLt` 当作 extended/grouped GEMM
的后端之一,被 PyTorch Inductor 通过 `ck4inductor` pip wheel 直接拉,被
`primus-turbo` 选择性使用。NVIDIA 侧对应物是 CUTLASS + 部分 cuBLASLt
实现的并集。后续蒸馏 `AITER` / `primus-turbo` / `hipBLASLt` 时,它们都
会反过来引用本文。

## 6. 进一步阅读 / TODO

入口文件(≤ 5,各一句"为什么读"):

- `README.md` + `TERMINOLOGY.md` —— 4 层架构图与术语表,任何后续阅读
  的锚点。
- `include/ck_tile/README.md` —— 唯一明确声明 "ck_tile 是新代码推荐
  入口、老 ck 处于迁移态" 的官方文档。
- `include/ck_tile/core/algorithm/coordinate_transform.hpp` —— coordinate
  transformation 抽象的真实形状(只看 primitive 列表与组合方式,不读
  实现)。
- `client_example/01_gemm/gemm.cpp` —— Client API 三件套最短模板。
- `tile_engine/operation_support_matrix.md` —— 当前 op × dtype × arch
  状态的权威表(读这一份就不用自己再统计了)。

待沉淀到别处的开放问题(**不在本文,继续做就归别的目录**):

- Pipeline `Problem`/`Policy`/`Scheduler` 三件套对应到我们 super-kernel
  设计的细节,归 `knowledge/kernels/cco-overlap.md` 与
  `.cursor/skills/cco-pipeline-overlap/SKILL.md`。
- `include/ck_tile/ops/fused_moe/` 与 `PrimusFP8GroupedMLP` 的差异,归
  `knowledge/kernels/fp8-expert-gemm.md` 或后续 MoE kernel 文档。
- `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` 等 MX 指令的
  per-lane register 布局,归 `.cursor/skills/mi355_hardware_aware/SKILL.md`
  与 `knowledge/hardware/`。
- `rocm_ck/` 的 constexpr Signature + kpack 多 arch 分发模式,是否给
  primus-turbo 的"单包多 arch"分发用 —— 归 `notes/` 实测后再说。
