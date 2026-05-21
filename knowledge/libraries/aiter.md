# AITER (AI Tensor Engine for ROCm)

> **Repo:** `ROCm/aiter` &nbsp; **Local path:** `3rd/aiter/`
> **Snapshot:** `163e6a025` &nbsp; `2026-05-21` &nbsp; on branch `main`
> **Size:** ~657 MB &nbsp; **License:** MIT
> **Distilled:** 2026-05-21

## TL;DR

AITER 是 AMD 官方的"production-ready 推理算子集合",定位是 vLLM /
SGLang / ATOM / JAX 在 ROCm 上的**默认 kernel 后端**(README 原文:"the
default kernel backend for LLM inference on AMD GPUs")。它本身不写
kernel —— 它的设计就是**一个统一的 op registry,把同一个 op 的 Triton /
CK / ck_tile / 手写 ASM 多种实现都收进来,按 (dtype, shape, arch) dispatch
到当前最优后端**;配套一套 JIT + on-disk cache,把 "怎么编、用哪条
backend" 这层复杂度从框架挪进库内。最值得借鉴的是**多后端 dispatch 模式**
与 **Opus 单 header HIP 模板库**。

## 1. 库定位 (positioning)

- **一句话:** AMD 的 production-grade 推理(并扩展到训练 + 通信)算子库,
  以"一个 op 多个 backend"的 dispatcher 形式把 CK / Triton / 手写 ASM 集
  成出统一接口,框架直接接入。
- **它是什么:**
  - 一套 PyTorch 友好的 op 集合(`aiter.fused_moe` / `aiter.mha` /
    `aiter.paged_attn` / 各种 `gemm_op_*` / RMSNorm / RoPE / 各种
    fused-quant op)。
  - 一个 JIT 构建系统(`aiter/jit/`),按需把每个 op 的 C++/HIP 源编译
    成 per-module `.so`,落到 on-disk cache(`~/.aiter/jit/build/`)。
  - 一个多后端 dispatcher,在 op 内部按 dtype / shape / arch / 环境变量
    选 Triton / CK / ck_tile / `.hsa/<arch>/*.co` 预编译 ASM code object 之
    一(README Key Features 原文:"Triton, Composable Kernel (CK), and
    hand-tuned ASM")。
  - 一个独立的轻量 HIP 模板库 **Opus**(`csrc/include/opus/`,单
    header),供库内和外部用户写自己的 HIP kernel,关键目标是 build-time
    最小化。
- **它不是什么:**
  - 不是图编译器(不做 fusion 搜索或 graph capture)。
  - 不是单一 kernel 库 —— 它本质是 "facade + dispatcher",底下大量
    kernel 来自 CK / ck_tile / hipBLASLt / Triton / 手写 ASM。
  - 不是 training-first(虽然 README 写明 "also training and
    GEMM+communication fused kernels",重心仍在推理 op 上)。
- **谁在用:** vLLM(ROCm 默认 attention 后端)、SGLang(ROCm Docker 默
  认)、ATOM(原生构建在 AITER 之上)、JAX-AITER(XLA FFI 桥)、若干客
  户自研推理引擎。

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| `aiter/ops/` | Python op surface,一个 op = 一个 module,内含 dispatch 与 backend 选择 | 与 `csrc/<backend>` 子目录一一对应,例如 `gemm_op_a8w8.py` 对接 `csrc/ck_gemm_a8w8/` |
| `aiter/jit/` | JIT 构建 + on-disk cache + lock 协议 | `core.py` 是入口;`@compile_ops(md_name, gen_func=...)` 装饰器把 Python 函数绑到 lazy-built `.so` |
| `aiter/configs/` | 各 op 的 tuned shape→config CSV 表 | autotuning pipeline 的输出归宿,被 dispatcher 在选 backend 时查 |
| `aiter/fused_moe*.py` + `aiter/paged_attn.py` + `aiter/mla.py` | 顶层 fused-op,典型多 backend dispatch 实例 | 同一个 fused_moe 内部按 dtype/quant scheme 切换 CK / opus / FlyDSL |
| `csrc/ck_*` / `csrc/cktile_*` / `csrc/py_itfs_ck/` | CK / ck_tile 后端胶水(包含 instance codegen) | 与 `3rd/composable_kernel` 的 Layer 4 Client API 对接 |
| `csrc/kernels/` | 库自带的 .cu/.hip 内核(quant / norm / rope / cache 等) | 大头是 ASM 不擅长、CK 又太重的中等粒度 fused op |
| `csrc/include/opus/` | 单 header HIP 模板库 + `hip_minimal.hpp` 子集 | 内部 kernel 与外部用户共享;build-time 是显式设计目标 |
| `hsa/<gfx942\|gfx950>/` | 手写 ASM 编出来的 `.co` code object 集合 | dispatcher 直接 `hipModuleLoad` 加载;覆盖 fmha / fmoe / allreduce 等热路径 |

## 3. 核心设计理念 (core design ideas)

### 3.1 同一 op 内的多后端 dispatch(Triton + CK + ck_tile + 手写 ASM)

README Key Features 原文:"Multiple kernel backends — Triton,
Composable Kernel (CK), and hand-tuned ASM"。这条 idea 不是把多个后端
并列"摆出来",而是**把 backend 选择吞进 op 自身**:同一个
`aiter.fused_moe` / `aiter.mha_fwd` / `gemm_op_a8w8` 内部按 (dtype,
shape, gfx, quant scheme, 环境变量) dispatch 到 CK instance、ck_tile
kernel、`hsa/<arch>/*.co` ASM code object 或 Triton kernel 之一。

这条选择回答的问题是:"客户的框架不希望知道 ROCm 上哪个 shape 适合哪条
backend"。AITER 把 "每条 backend 的 sweet spot" 这种领域知识收编到库
内(典型例子:`aiter/fused_moe.py` 里 CK sorting 与 Opus sorting 在不同
路径下被选用、`AITER_USE_CK_MOE_SORTING` 环境变量做 override),框架方
只 `import aiter; aiter.xxx(...)` 即可,不需要知道下面是哪个 kernel
flavor。

### 3.2 JIT 编译 + on-disk cache + 多进程 file-lock

`aiter/jit/core.py` 的设计:每个 op 模块声明一个 `md_name`(例如
`module_aiter_mha_fwd_bf16`),`@compile_ops` 装饰器在首次调用时触发
`build_module(md_name, srcs, flags_extra_cc, flags_extra_hip, blob_gen_cmd,
...)`,产出 `~/.aiter/<pkg>/build/<md_name>/.../module_xxx.so` 并 import。
后续进程命中现成 `.so` 直接跳过编译。多进程并发用 `FileBaton`
(`utils/file_baton.py`)做 lock,避免重复编译。环境变量 `AITER_REBUILD`、
`AITER_JIT_DIR`、`ENABLE_CK`、`AITER_ENABLE_EXPERIMENTAL`、
`AITER_FP4x2`、`GPU_ARCHS` 等是这套 JIT 系统的公开 knob。

这条选择回答的是:"AITER 想覆盖 10+ dtype × 几十种 quant scheme × 多
arch,但 CK 那种 AOT 全枚举对终端用户太重"。所以 AITER 把 "枚举 +
build" 推到运行期首次调用,代价是首调延迟,换来安装包小、按需 build、
按 (dtype, shape) 子集化模板特化。与 CK 的 AOT instance 库(见
[`composable-kernel.md`](composable-kernel.md))是互补关系:CK 提供模板,
AITER 在 JIT 时把模板特化到当前 op 形状然后落 cache。

### 3.3 Production op registry —— 默认后端身份带来的设计约束

README 原文:"the default kernel backend for LLM inference on AMD
GPUs",`Ecosystem` 表点名 vLLM / SGLang / ATOM / JAX 都已经接入。这不是
营销,而是直接决定了 AITER 的设计取舍:

- **接口稳定性高于内部漂亮度**:同一个 `aiter.fused_moe_xxx` 的签名一旦
  发布,内部即便换 backend 也必须保持外部行为不变;tuned CSV
  (`aiter/configs/`)也是兼容性资产之一。
- **支持矩阵以 production 模型为单位**:News 区直接把 "MI355X tuned
  configs for Kimi-K2.5 and DeepSeek-V3" 当成 release 卖点,说明配置是按
  目标模型反向调优、不是按算子单独调优。
- **集成路径是"框架直接 import"**:`aiter/ops/aiter_operator.py` 用
  `@compile_ops` 把 `aiter.add` / `aiter.sub` / `aiter.sigmoid` 直接暴露成
  PyTorch 风格的 op,框架方写 `import aiter; aiter.mha_fwd(...)` 就行。
- **autotuning 入 CI**(`docs/autotuning_pipeline.md`):tuned CSV 是
  仓库版本化资产,CI pipeline 周期性 sweep → 更新 CSV → 后续推理直接命
  中。

### 3.4 Opus —— 独立的单 header HIP 模板库(`csrc/include/opus/`)

`csrc/include/opus/README.md` 自述:"Inspired by ck/ck_tile and
cutlass/cute, opus adopts a significantly simplified design while
prioritizing maintainability. Distributed as a single-header library
(`opus.hpp`), opus provides only essential abstractions ... positions
itself above hand-written HIP kernels yet below highly optimized
template libraries like ck/cutlass."

定位本身就把 Opus 与 CK 区分开:Opus 不追求"一份模板覆盖所有
op",只提供 (a) AMDGPU dtype + 转换、(b) 自动 vectorized load/store
(`make_gmem`)、(c) 少量 layout/adaptor、(d) MFMA 包装。**核心设计目标是
build time**:Opus README 那一节 "Best Practice to Reduce HIP Kernel
Compile Times" 给出一套可落地的优化(用 AMDGCN builtin 替换
`<hip/hip_runtime.h>`、`-D__HIPCC_RTC__` 关 stdlib 隐式 include、
`--genco` 走 device-only 编译 + `hipModuleLaunchKernel` 取代 torch
extension、`__HIP_DEVICE_COMPILE__` guard 重头模板),声称对
`warp_sort_bitonic` 这种小 kernel 可达 **61× 编译加速**(21 s → 346
ms)。它存在的意义是:**当 CK 的模板太重、又不想退回纯 HIP**,Opus 是
那个中间层。

### 3.5 多模实现共存:`hsa/<arch>/*.co` ASM code object 是一等公民

`hsa/gfx942/` 与 `hsa/gfx950/` 下放着大量 `.co` 文件(`fmha_v3_fwd/`,
`fmoe_*`,`allreduce_*`,`bf16gemm/`,`f4gemm/` 等),是手写汇编经
`hsa/codegen.py` + `hipcc` / `clang` 流水线产出的 device code object。
Dispatcher 在 op 内部按 (gfx, shape, dtype) 决定到底是调 CK kernel 还是
`hipModuleLoad` 这些 `.co`。

设计含义:AITER 接受**对最关键的热路径(MLA decode / FMHA / fused MoE /
allreduce + RMSNorm)直接维护手写汇编**的工程成本,因为 README 的性能
数字(MLA decode 17× / MHA prefill 14× / DeepSeek-R1 e2e 2.1×)主要从
这一层挤出来 —— 高层模板库的可移植性在这里让位于 ASM 级的硬件 utilization。
这是 CK 显式不做、framework 又做不了的位置。

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| **多后端 dispatch 收进 op 自身**(同一个 Python op,内部按 dtype/shape/arch 选 CK / Triton / hipBLASLt / 手写 ASM) | 上层框架不必知道 "什么 shape 走什么 backend",但要保留"出现更快的 backend 时无缝替换"的权利 | `primus-turbo` 的 `PrimusFP8GroupedMLP` 当前在 Python 里硬编一条 Triton 路径,可以仿照 AITER 把 hipBLASLt grouped GEMM、CK ck_tile grouped GEMM、Fused Triton grouped GEMM 收进同一个 autograd.Function,内部按 (E, tokens_per_expert 直方图, dtype, gfx) dispatch;`fp8-expert-gemm.md` 那张 "为什么 280 ms 差距来自模块框架而非 GEMM" 的结论会更稳 | dispatcher 自己有 overhead,需要给热路径准备 fast-path(命中后按指针 cache 跳过 dispatch);backend 选择规则容易变成隐式知识,要写下来 |
| **JIT cache key 设计**(`md_name` = op + dtype suffix,`gen_func` 把 dtype/shape 折进 `blob_gen_cmd`,落到 `~/.aiter/jit/build/`,多进程用 `FileBaton` lock) | 训练时如果要用 JIT'd kernel(MoE permute、fused fp8 quant 等),首调编译 + 多 rank 并发 + 多节点共享都是坑 | 我们在 Primus 训练栈 ship JIT kernel(无论是 Triton 还是 HIP)时,可以照搬这套约定:cache 路径在共享存储、key 包含 (gfx, ROCm version, dtype, tile shape, env hash)、并发用 `FileBaton`,避免几十个 rank 同时 hipcc | 多节点 SLURM 跑要么共享 cache 要么预热,否则 step-0 会被首调延迟撕开;cache key 漏掉 hipcc flag 会导致 silent ABI mismatch |
| **Op registry + Python-level autograd 包装,与 primus-turbo 的对比** —— AITER 用 `@compile_ops` 把每个 op 暴露成 `aiter.<op>(...)`,autograd 留给上层 framework 自己包;primus-turbo 走另一条路,kernel 入口直接是 `torch.autograd.Function`(`primus_turbo/pytorch/ops/gemm_fp8.py`、`gemm_fp4.py`、`normalization.py` 等都按 Function 拆) | "kernel 库" vs "training fused-op 库"这两层职责应该不应该合并 | 给 primus-turbo 设计新的 op 入口时,可以借 AITER 那一层 thin op surface:`primus_turbo.<op>(...)` 暴露 forward-only kernel,`primus_turbo.fn.<op>` 才是 autograd.Function。这样 (a) op 可以被 graph compiler / Inductor 当成 custom op 注册,(b) `PrimusFP8GroupedMLP` 这种重模块能直接复用底层 kernel 而不必各自再包一遍 | 边界要划清:training 才需要的 weight cache、tokens_per_expert D2H sync 这些 stateful 行为不能塞进 thin op 层,否则 inference 端会重新付一遍 |
| **Opus 风格的单 header HIP 模板库 + build-time-first 设计哲学**(`opus.hpp` + `hip_minimal.hpp` + `--genco` + `ctypes.CDLL`) | 我们自写 HIP 工具 kernel 时,torch extension 编译 8 s 是开发体验的最大杀手 | `primus-turbo` 的 HIP 子目录、MonolithMoE super-kernel 的工具头、Pilot 自动 sweep 出的中间 kernel,可以做一个 `primus_hip_micro.hpp` 单 header,把 dtype 转换 + vectorized load/store + MFMA 包装收进去,配合 `__HIP_DEVICE_COMPILE__` guard 与 ctypes,把单 kernel 编译时间从 ~10 s 压到 < 1 s,让 Pilot 的 sweep 真正可用 | 单 header 重 template 容易让二级编译时间反弹,需要约束模板深度并显式 `extern template`;ctypes 入口失去 torch dispatch / autograd,只适合 forward-only 工具 |
| **预编译 ASM `.co` 作为 dispatcher 一档**(`hsa/<arch>/*.co` + `hipModuleLoad`) | 编译器无法压榨干净的极热路径(MLA / FMHA / 部分 fused MoE / 通信 + norm)需要手写,但又要走统一 op 入口 | 我们 MonolithMoE 的 super-kernel 一旦稳定下来,可以走同一条路:汇编源跟 codegen 脚本入仓、按 (gfx, problem-class) 编出 `.co`、Python 端 `hipModuleLoad` 拉,作为 dispatcher 的最热档(命中率低但延迟最低) | ASM 路径与 ROCm 版本耦合极紧,需要每个 LLVM bump 都过一次回归;`.co` 入仓会让仓库膨胀,要考虑 LFS / 外置 release |
| **Autotuning 入 CI + tuned CSV 入仓**(`aiter/configs/*.csv` + `docs/autotuning_pipeline.md` 的两条 pipeline) | 算子 tuning 结果如果只在每个工程师的本地,产品迭代时永远在重新踩坑 | Primus 这边目前的 tuning 数据(MoE tile shape、grouped GEMM block 形状、cco overlap 的 chunk size)散落在 notes/,可以学 AITER 把 (shape → 最优 config) CSV 化、放仓库、由 CI 周期性 sweep 更新 | 入仓的 CSV 与硬件 + ROCm 版本强耦合,要在 CSV header 里固化这些维度;sweep 自身是大开销,要分级(level0/1/2)而不是每 PR 全跑 |

## 5. 与生态的关系 (ecosystem position)

```
PyTorch / vLLM / SGLang / ATOM / JAX
        │
        ▼
    AITER  (op registry + JIT + dispatcher)
        │
        ├── Triton          (norms / 部分 attention / 通信)
        ├── CK / ck_tile    (GEMM 主力,见 knowledge/libraries/composable-kernel.md)
        ├── hipBLASLt       (部分 extended GEMM)
        ├── 手写 ASM       (.co code object,fmha / fmoe / allreduce 热路径)
        └── Opus           (库内与外部用户的轻量 HIP 模板,build-time-first)
```

AITER 在 AMD 软件栈里是"贴框架"的那一层:**上**对接 vLLM /
SGLang / ATOM / JAX,**下**复用 `3rd/composable_kernel` 的 ck_tile
templates(GEMM 主力)、`Triton on ROCm`(norm / 通信)、`hipBLASLt`
(部分 GEMM)、库内的手写 ASM(`hsa/<arch>/*.co`)。它和
[`composable-kernel.md`](composable-kernel.md) 描述的 CK 是**互补**而非
竞争:CK 提供模板与 AOT instance,AITER 在 op 边界做 JIT 特化 + tuned
config + 多后端 dispatch + production API 稳定性。Opus 则是 AITER 自己
喂养的"我们想要 ck_tile 那种抽象,但不能接受它的编译时间和模板深度"的
回答。NVIDIA 侧没有完全等价物;最接近的是 cuBLASLt + cuDNN + 部分手写
ASM(SASS)的并集,但缺一个统一 op registry。

## 6. 进一步阅读 / TODO

入口文件(≤ 5,各一句"为什么读"):

- `README.md` —— 唯一明确写出 "default kernel backend for LLM inference
  on AMD GPUs" 与 "Triton + CK + ASM" 设计取向的官方文档。
- `aiter/jit/core.py` —— JIT 系统全部入口在这里(`compile_ops`、
  `build_module`、`get_user_jit_dir`、`FileBaton`),想抄 cache 协议从
  这看。
- `csrc/include/opus/README.md` —— Opus 的定位 + build-time 优化清单,
  想做单 header HIP 工具库必读。
- `aiter/fused_moe.py` + `aiter/ops/moe_op.py` —— 多后端 dispatch
  在真实 op 里长什么样(CK sorting vs Opus sorting 的 toggle 是典型样
  本)。
- `docs/autotuning_pipeline.md` —— tuning CSV 入 CI 的工作流,想把
  primus 这边的 tuning 数据沉淀也走同一条路时可参考。

待沉淀到别处的开放问题(**不在本文,做了归别的目录**):

- AITER 的 op dispatcher 在每个 op 内的具体规则(哪些 (dtype, shape, gfx)
  走 CK / Triton / ASM),归 `notes/` 单独实测一次后总结,不归这里。
- AITER 与 `primus-turbo` 的 fused-op 集合差异(命名、forward-only vs
  autograd、weight cache 模式),等 `primus-turbo.md` 蒸馏完后,在
  `knowledge/libraries/_patterns.md` 里做横切对比。
- Opus 的 MFMA wrapper 与 ck_tile 的 `WarpGemmAttributeMfmaImpl_*` 风格
  对比,归 `knowledge/kernels/` 的某个 MFMA pattern 文档。
- ASM code object(`hsa/<arch>/*.co`)的 codegen pipeline(`hsa/codegen.py`)
  是否能给我们 MonolithMoE 的 super-kernel 发布流程复用,归
  `.cursor/skills/cco-pipeline-overlap/` 与 `notes/` 实测决定。
