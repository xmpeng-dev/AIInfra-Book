# Primus-Turbo

> **Repo:** `AMD-AGI/Primus-Turbo` &nbsp; **Local path:** `3rd/turbo/`
> **Snapshot:** `f9a37d9` &nbsp; `2026-05-21` &nbsp; on branch `main`
> **Size:** ~11 MB &nbsp; **License:** MIT
> **Distilled:** 2026-05-21

## TL;DR

Primus-Turbo 是 AMD AGI 的**训练侧 fused-op 库**,在 Primus 产品矩阵里
和端到端框架 Primus-LM、稳定性平台 Primus-SaFE 三足鼎立。它的设计选择
非常明确:**不自己写 kernel,而是站在 CK / hipBLASLt / AITER / Triton
之上,做统一的 op surface + autograd 包装 + 多后端 dispatch**。核心抽象
是 `BackendType` 枚举(CK / HIPBLASLT / AITER / TRITON / DEEP_EP /
TURBO)+ `AutoKernelDispatcher`(per-(op, precision, shape) profile-and-
cache),配上 `primus_turbo.pytorch` / `primus_turbo.jax` 两个并列子包,
以及 FP8 / FP4 / GroupedGEMM / DeepEP 这套训练特化的 op 组合。最值得借
鉴的是"vendor-lib 之上做 dispatcher + Function"这套薄壳模式。

## 1. 库定位 (positioning)

- **一句话:** AMD ROCm 上的训练侧 fused-op 库,把 CK / hipBLASLt /
  AITER / Triton 的算子统一封装成 PyTorch / JAX 友好的 API,服务 Primus-
  LM 与上游训练框架(Megatron / TorchTitan)。
- **在 Primus 矩阵里的角色:**(README "Primus Product Matrix" 表)
  - **Primus-LM** — E2E 训练框架,跑在 Megatron / TorchTitan 之上。
  - **Primus-Turbo**(本库)— 高性能算子与模块,核心特征 "Integrates
    multiple high-performance backends (e.g., CK, hipBLASLt, AITER)"。
  - **Primus-SaFE** — 集群稳定性 + 拓扑感知调度 + fault tolerance。
- **它是什么:**
  - 一套 PyTorch / JAX 双前端的训练算子库(`primus_turbo.pytorch` /
    `primus_turbo.jax`,通过环境变量 `PRIMUS_TURBO_FRAMEWORK=PYTORCH|JAX`
    选编译目标)。
  - 一套训练特化的 op:GEMM / GroupedGEMM / FP8 GEMM / FP4 GEMM /
    FlashAttention / RMSNorm / MoE permute / DeepEP all-to-all / fused
    async-TP 等。
  - 一套显式的 backend 抽象层(`primus_turbo.pytorch.core.backend`):
    `BackendType` 枚举 + `KernelBackend` 抽象基类 + `AutoKernelDispatcher`
    profile-and-cache,**没有 JIT、没有 codegen,只做 dispatch 与编排**。
- **它不是什么:**
  - 不是 kernel 库 —— GEMM / Attention / GroupedGEMM 的实际实现都来自
    CK / hipBLASLt / AITER / Triton;Primus-Turbo 自己只在 `csrc/kernels/`
    维护少量轻量 fused kernel(quantization / normalization / moe_permute
    /shuffle / reduce 等)。
  - 不是推理引擎 —— 与 AITER(推理 default backend)显式分工,DeepEP
    `README.md` 里写明 "only used for training and doesn't support
    low-latency kernels"。
  - 不是 framework —— 没有 dataloader / scheduler / checkpoint 这套,
    那些归 Primus-LM。
- **谁在用:** Primus-LM(深度集成)、Megatron-LM / TorchTitan 通过
  Primus-LM 间接消费、ROCm 训练 blog 里的 MoE 训练 best practice 直接挂
  在本库上。

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| `primus_turbo/pytorch/` | PyTorch 前端的 op surface + autograd.Function 包装 + nn.Module | 与下方 `jax/` 完全平行,**接口对称** |
| `primus_turbo/jax/` | JAX 前端,通过 XLA FFI(`jax.ffi.register_ffi_target`)绑底层 kernel | `__init__.initialize()` 注册一组 IMPL/ABSTRACT/LOWERING table |
| `primus_turbo/triton/` | 库内 Triton kernel(各 op 的 Triton backend 实现) | 与 `pytorch/ops` 通过 `kernels/<op>_impl.py` 接 |
| `primus_turbo/common/` | 跨前端共享:常量(全部 `PRIMUS_TURBO_*` 环境变量集中在 `constants.py`)、logger | 唯一的 cross-frontend 共享代码 |
| `primus_turbo/pytorch/core/backend.py` | **dispatcher 中枢**:`BackendType` enum、`KernelBackend` ABC、`GlobalBackendManager`、`AutoKernelDispatcher`、`TuneCache` | 一切多后端选择在此 |
| `primus_turbo/pytorch/{ops,modules,kernels}/` | ops:autograd.Function 表层;modules:`Float8Linear` 等 nn.Module;kernels:每个 op 一组 backend 实现类 | 三层分得很干净 |
| `primus_turbo/pytorch/deep_ep/` | DeepEP all-to-all(MoE EP 通信),依赖 rocSHMEM,实验态 | gfx942 only,显式标 "only used for training" |
| `csrc/` | C++/HIP 绑定 + 少量自带 fused kernel(`csrc/kernels/{deep_ep,gemm,grouped_gemm,moe_permute,normalization,quantization,reduce,shuffle}`) | 真正算子实现的大头在 `3rdparty/composable_kernel/` |
| `3rdparty/composable_kernel/` | submodule:CK 源码作为构建依赖 | 见 [`composable-kernel.md`](composable-kernel.md) |

## 3. 核心设计理念 (core design ideas)

### 3.1 训练侧定位:与 AITER 在产品矩阵上显式分工

README 的 Primus Product Matrix 表里,Primus-Turbo 的 Role 写得很直接:
"High-performance operators & modules ... core training operators and
modules (FlashAttention, GEMM, GroupedGemm, DeepEP etc.)"。这条与
[`aiter.md`](aiter.md) 的 "production default for **LLM inference** on
ROCm" 形成正面对照 —— 同样是 AMD 的算子集合,AITER 服务 vLLM / SGLang
推理路径,Primus-Turbo 服务 Megatron / TorchTitan 训练路径。

这条定位带来的具体设计后果:**op surface 以 autograd.Function 为单位**
(`primus_turbo/pytorch/ops/gemm_fp8.py` 的 `FP8GemmTensorFunction` /
`FP8GemmRowFunction` 等都是 `torch.autograd.Function`);**精度组合以
training 用得到的为主**(FP8 hybrid E4M3 fwd + E5M2 bwd 是默认,见
`Format.HYBRID` 在 `gemm_fp8.py` 的处理);**通信端拉进了 DeepEP**(MoE
EP 训练 must-have,推理用不到);**没有 KV cache / paged attention**
(那些归 AITER)。

### 3.2 站在 vendor lib 之上,而不是重写 kernel

`primus_turbo/pytorch/core/backend.py` 的 `BackendType` 枚举把可选实现
全部摆出来:`CK / HIPBLASLT / AITER / TRITON / DEEP_EP / TURBO`。其中
`TURBO` 才是库自己的 fused kernel,其余四档全是引入的上游(CK 走
submodule、hipBLASLt 走 ROCm 自带、AITER 走 PyPI、Triton 走 ROCm 自带)。
每个 op 的 `kernels/<op>_impl.py` 里给每种 backend 写一个
`KernelBackend` 子类(例:`GEMMFP8HipBLASLtBackend` 在
`kernels/gemm/gemm_fp8_impl.py`),实现 `can_handle(**kwargs)` 与
`execute(**kwargs)` 两件事。

这条选择的好处直白:每条 backend 上 ROCm 都已经投入了独立优化人月,
Primus-Turbo 把它们**统一到一个 op surface 下**,训练 stack 不必为
"何时用 CK / 何时用 hipBLASLt / 何时用 Triton" 各自接管;代价是 backend
之间的 dtype × layout × shape 支持差异要在 `can_handle` 里抹平,新加
backend 要补全所有 op,且**任一上游回归都会直接打到训练 stack**。这是
Primus-Turbo 与"自己从零写 kernel"路线(例如 NVIDIA TransformerEngine
更多自有 CUTLASS 实现)最显著的区别。

### 3.3 多前端包结构:`primus_turbo.pytorch` / `primus_turbo.jax` 并列

`primus_turbo/{pytorch,jax}/` 两个子包接口对称,各自走自己框架的扩展
机制:PyTorch 走 `torch.library.custom_op` + `torch.autograd.Function`,
JAX 走 `jax.ffi.register_ffi_target` + 一组 `IMPL_TABLE /
ABSTRACT_EVAL_TABLE / LOWERING_TABLE / TRANSPOSE_TABLE`(`primus_turbo/
jax/__init__.py` 的 `initialize()`)。共享层薄到只剩 `primus_turbo/
common/`(常量、logger);**没有强行抽出 framework-neutral op
abstraction**。

构建端用 `PRIMUS_TURBO_FRAMEWORK=PYTORCH|JAX` 在安装时切到目标前端
(README "Development" 段),产出的 wheel 只编一边。设计含义:**两个
前端共享底层 kernel C++/HIP 源(`csrc/kernels/`),但 Python op surface
各自独立演化**,不再为了"接口统一"付出抽象成本。当前 PyTorch 路径成
熟,JAX 标 "under active development"(README 顶部 Note)。

### 3.4 FP8 训练专属设计:Format / Granularity / Function 三件套

`primus_turbo/pytorch/core/low_precision.py` 把 FP8 训练的可调维度收成
三个类型:**`Format`**(E4M3 / E5M2 / HYBRID)、**`ScalingGranularity`**
(TENSORWISE / ROWWISE / BLOCKWISE / MX_BLOCKWISE)、**`Float8QuantConfig`**
把两者打包。`FP8GemmTensorFunction` / `FP8GemmRowFunction` /
`FP8GemmBlockwiseFunction` / `FP8GemmMXBlockwiseFunction` 等
autograd.Function 按 granularity 拆;`Float8Linear` 把它包成 nn.Module。
HYBRID 在 `forward` 用 E4M3、`backward` 用 E5M2 的 fwd/bwd 异构 dtype 在
`_get_fp8_dtype(format, is_fwd_stage)` 一行写完。

它**没有**实现"权重 FP8 cache"这种 stateful 模块 —— 那个责任留给上层
(`PrimusFP8GroupedMLP` 在 Primus-LM 一侧,见
[`../kernels/fp8-expert-gemm.md`](../kernels/fp8-expert-gemm.md))。
Primus-Turbo 的边界明确卡在"forward-only op + autograd 包装"这一层,
weight-cache、tokens_per_expert D2H sync、模块级 framework overhead 不
进库。这条边界本身就是设计:**op 库不该 stateful,有状态的训练逻辑
属于 framework adapter**。

### 3.5 `AutoKernelDispatcher`:per-shape profile-and-cache 的 backend 选择

`primus_turbo/pytorch/core/backend.py:AutoKernelDispatcher.dispatch()`
的优先级写得很直白(也写在 `GlobalBackendManager` 的 docstring 里):
**(1) user code `set_*_backend()` → (2) 环境变量
`PRIMUS_TURBO_{GEMM,GROUPED_GEMM,MOE_DISPATCH_COMBINE}_BACKEND` → (3)
auto-tune(`PRIMUS_TURBO_AUTO_TUNE=1` 时实测所有 `can_handle` 的 backend
取最快,落 `TuneCache` LRU)→ (4) code default → (5) fallback 遍历所有
backend**。

环境变量的解析允许 per-precision 配置(格式
`FP8:HIPBLASLT,FP4:AITER,OTHER:CK`,`_extract_backend_from_env` 处理),
这把"FP8 走 hipBLASLt、FP4 走 AITER、其余走 CK"这种实际经常出现的偏
好编进了配置语义。Auto-tune 会在 `cuda graph capture` 中跳过(避免抖
动),`TuneCache` 满会 warn "shapes changing frequently — AutoTune may
not be beneficial"。这条机制比起把"如何选 backend"散落在每个 op 里要
清爽得多,也是本库最直接可以借鉴出去的中枢。

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| **"站在 vendor lib 之上"的薄壳模式**(`BackendType` 枚举 + `KernelBackend` ABC + `AutoKernelDispatcher`,**不自己写 kernel** 是核心约束) | 内部每个团队不可能都把 GEMM / Attention 写得比 AMD ROCm 团队更好,但又需要统一 op 入口 | 我们如果给 Primus 起一个新的内部算子层(比如 MoE 优化套件),应当默认从 CK / AITER / hipBLASLt / Triton 里挑,而不是默认手写 HIP;**只有 dispatcher 命中不到任何 backend 才落 `TURBO` 自有实现** | 任一上游(CK 模板 / AITER JIT / hipBLASLt ABI)回归都会直接进 Primus-LM,需要把上游版本 pin 在 submodule + version-lock,并在 CI 里跑 dispatcher fallback 覆盖率 |
| **FP8 weight-cache 模式 = "无状态 op 在库内 + 有状态 cache 在 framework adapter"**(Primus-Turbo 自己只暴露 `gemm_fp8` / `grouped_gemm_fp8` forward,weight FP8 cache 在 `PrimusFP8GroupedMLP` 一层) | 训练里 FP8 weight 重复 quantize 是性能瓶颈,但 cache 一旦下沉到 op 库就破坏 op 无状态性 | 这个边界已经被 [`../kernels/fp8-expert-gemm.md`](../kernels/fp8-expert-gemm.md) 验证过 —— 端到端 280 ms 差距里 277 ms 来自模块框架而非 GEMM。后续做新的低精度训练算子(FP4 / MX FP8)时,把"算子 forward"和"weight cache + 量化 schedule"两层在代码上严格分离,cache 一律放 `primus/backends/megatron/core/extensions/` 而不是 `primus_turbo` 内 | weight-cache 的 invalidation 协议(每多少 step 重 quant、optimizer step 后失效)是 framework 侧的责任,需要写明 ownership;否则 op 库会被 PR 推着加 stateful flag |
| **Training-vs-inference op surface 划分**(Primus-Turbo 训练 / AITER 推理,在产品矩阵层面就分干净) | 训练和推理对算子的需求差异(autograd / GroupedGEMM / DeepEP all-to-all vs paged attention / KV cache / sampling)塞进同一个库会让 dispatcher 选项爆炸 | 我们自己规划任何 op 库时,先回答"训练还是推理"再决定 op 集合与 backend 矩阵 —— 例如要给推理侧加 MLA decode,默认应当走 AITER 而不是 Primus-Turbo;给训练侧加 GroupedGEMM FP4,默认应当走 Primus-Turbo 加新 backend 而不是 AITER | 灰区是"两边都用得到的 op"(GEMM / RMSNorm / RoPE),两边都包一遍是显式的工程成本,但比硬塞进一个库要清楚 |
| **多前端并列子包**(`primus_turbo.pytorch` / `primus_turbo.jax` 接口对称、共享只到 `common/` 这一层,通过 `PRIMUS_TURBO_FRAMEWORK` 在构建期切) | 强行抽 framework-neutral op API 会牺牲两边的原生扩展机制(PyTorch 的 autograd / JAX 的 FFI + primitive) | 我们自己如果未来要给一个算子加 JAX 入口或反过来,**不要强求一份 Python 实现两边跑**,直接照 Primus-Turbo 这样起一个 `mylib.jax/` 平行包,共享 `csrc/kernels/`,在构建期分发 | 接口"对称但不共享"靠 review 守住,容易漂移;需要 lint 或 schema 检查保证两边 op 名 / 参数顺序一致 |
| **显式的 `BackendType` 枚举 + 环境变量 per-precision 解析**(`PRIMUS_TURBO_GEMM_BACKEND=FP8:HIPBLASLT,FP4:AITER,OTHER:CK`) | 多 backend 选择真实使用时往往按 precision 而不是按 op 切 | 我们自己的 dispatch 配置(目前散落在 Primus run.sh / 各 fused-op 的 env-var 集合)可以借这套语义:**一个变量 + per-precision 段**,避免 `PRIMUS_FP8_USE_HIPBLASLT` / `PRIMUS_FP4_USE_AITER` 这种逐 op 变量蔓延 | 解析逻辑要写测试(典型 corner case:precision 段缺省、`OTHER` 占位、大小写),否则配置漂移 silent 失败 |
| **`AutoKernelDispatcher` 的 profile-and-cache + LRU + graph-capture skip**(`TuneCache` LRU,满了 warn,`is_current_stream_capturing()` 时跳过 tune) | per-shape 实测 best backend 与训练实战的两个坑:形状不稳定(LRU 爆) + cuda graph 抖动 | 我们在 cco super-kernel / Pilot 自动 sweep 这两条路线上做 best-of-N 时,可以直接借这个类 —— 把 (shape, dtype, gfx) 作为 key,实测落 LRU,graph capture 时跳过 | warmup_iters=10 / profile_iters=20 是 Primus-Turbo 的默认,跨 op / 跨 backend 不通用,需要按 op 调;profile 阶段会污染 step-0,需要前置 warmup |

## 5. 与生态的关系 (ecosystem position)

```
Primus-LM (Megatron / TorchTitan)
        │
        ▼
   Primus-Turbo  (op surface + AutoKernelDispatcher + autograd.Function / JAX primitive)
        │
        ├── CK / ck_tile     (GEMM / GroupedGEMM 主力,见 knowledge/libraries/composable-kernel.md)
        ├── hipBLASLt        (extended/grouped GEMM,FP8 路径默认)
        ├── AITER            (部分 op,见 knowledge/libraries/aiter.md)
        ├── Triton (ROCm)    (各 op 的 Triton backend)
        ├── DeepEP / rocSHMEM (MoE EP all-to-all 通信,gfx942 only)
        └── 自带 csrc/kernels/ (quantization / normalization / moe_permute / shuffle / reduce 等轻量 fused)
```

Primus-Turbo 在 AMD 训练栈里是**最接近 PyTorch/JAX 的那一层**:**上**
被 Primus-LM 通过 `primus/backends/megatron/core/extensions/primus_turbo.py`
这类适配层调用,**下**复用
[`composable-kernel.md`](composable-kernel.md) 描述的 CK ck_tile 模板与
AOT instance、[`aiter.md`](aiter.md) 描述的 AITER op registry、hipBLASLt
extended GEMM、ROCm Triton。

与 AITER 在产品矩阵层面**显式分工**(训练 vs 推理),但在实现层面**有
显著重叠**(同一个 FP8 GEMM 形状两边可能都有 backend),这种分工 +
重叠的结构是产品而非工程取舍的产物 —— 团队边界清晰,但用户层面要
靠文档约定 + dispatcher 配置守住"训练别去 import AITER、推理别去
import Primus-Turbo"。NVIDIA 侧最近的对应物是 TransformerEngine
(训练 fused-op 库),区别在于 TransformerEngine 更多走"自有
CUTLASS 实现",而 Primus-Turbo 是"vendor-lib dispatcher"。

## 6. 进一步阅读 / TODO

入口文件(≤ 5,各一句"为什么读"):

- `README.md` —— 唯一明确写出 Primus 产品矩阵(LM / Turbo / SaFE)
  与"Integrates CK + hipBLASLt + AITER"定位的官方文档。
- `primus_turbo/pytorch/core/backend.py` —— dispatcher 中枢全部抽象
  (`BackendType` / `KernelBackend` / `GlobalBackendManager` /
  `AutoKernelDispatcher` / `TuneCache`),想抄这套模式从这一个文件入手
  够了。
- `primus_turbo/common/constants.py` —— 全部 `PRIMUS_TURBO_*` 环境
  变量集中在一个文件,知识等于半本 user manual。
- `primus_turbo/pytorch/ops/gemm_fp8.py` + `primus_turbo/pytorch/
  modules/linear_fp8.py` —— FP8 training op + nn.Module 的典型样本,
  Function / Module 分层在这里看最清。
- `primus_turbo/jax/__init__.py` —— JAX 子包如何接 XLA FFI 与
  primitive table,要做多前端时的参考模板。

待沉淀到别处的开放问题(**不在本文,做了归别的目录**):

- Primus-Turbo `AutoKernelDispatcher` 在真实训练 step 的 best-backend
  分布(FP8 GEMM 实际命中 hipBLASLt 还是 CK?MoE EP 实际命中 DEEP_EP 还
  是 TURBO?),归 `notes/` 实测后归档。
- `csrc/kernels/moe_permute/` 与 AITER `aiter/fused_moe.py` 的 permute
  实现差异,归 `knowledge/kernels/` 的 MoE permute 专文(暂无)。
- DeepEP / rocSHMEM 在 gfx950 上的可用性(目前 `deep_ep/README.md` 明
  确仅 gfx942),归 `notes/` 跟踪。
- Primus-Turbo backend dispatch 行为与 Primus-LM 内 `_use_fused_fp8_triton`
  / `_use_hipblaslt_fp8` 等环境变量的关系(`knowledge/kernels/fp8-
  expert-gemm.md` 已记录这条 framework-layer dispatcher),后续如果两
  层 dispatcher 出现冲突,在 `knowledge/libraries/_patterns.md` 横切
  对比时一起总结。
