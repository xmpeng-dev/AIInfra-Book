# 第三方算子库:跨库设计模式综述

> **Distilled:** 2026-05-21
> **Scope:** 综合 `composable-kernel.md` / `aiter.md` / `primus-turbo.md` /
> `hipblas.md` 四份单库蒸馏,提炼 ROCm 算子栈里反复出现的设计模式,
> 并给出"未来做新算子时该选哪条路"的决策表。
> **不是** 任一单库的再蒸馏(那归 `distill-operator-repo` skill);
> **是** 跨库横切的工程取舍记录。

## TL;DR

ROCm 算子栈的四个代表性库共同回答了同一个问题:**当 GEMM / Attention /
MoE 的最优实现散落在 CK 模板、hipBLASLt extended API、Triton DSL、手写 ASM
和遗留 BLAS 五条路径上时,如何让框架方只用一个 op 入口就拿到当下最优的
那一档**。CK 给出 "模板 + AOT instance 枚举" 的基础设施层,hipBLAS 给出
"public API 不变、后端可互换" 的可移植模板,AITER 与 Primus-Turbo 在它们
上面各自做 inference 与 training 的 op registry + dispatcher,**整个栈
本质是一棵"vendor lib 之上叠 dispatcher"的依赖树,不是一组互相竞争的
kernel 集合**。最重要的横切结论是:我们自己写的任何新算子层,默认应当
**站在这棵树上**而不是平行重写;只有 dispatcher 落不到任何已有 backend
时,才该退到 "TURBO 自有 kernel"。

## 1. 库谱系图 (library DAG)

下图按 "framework / fused-op layer / kernel-template layer / 物理实现层"
分四层。**左半训练栈,右半推理栈,中下层共享同一组底层 kernel lib**。

```
                framework / product layer
       ─────────────────────────────────────────────
       [training stack]              [inference stack]
       Megatron / TorchTitan         vLLM / SGLang / ATOM / JAX
                │                                │
                ▼                                ▼
        Primus-LM (E2E)                      (framework 自有 runtime)
                │                                │
                ▼                                ▼
        Primus-Turbo                          AITER
   (autograd.Function 表层 +              (op registry + JIT +
    BackendType + AutoKernelDispatcher)    多后端 dispatcher)
                │                                │
                │   ┌────────────────────────────┤
                │   │                            │
                ▼   ▼                            ▼
            CK / ck_tile                    手写 ASM
            (Layer 1–3:                    (hsa/<arch>/*.co,
             tile primitives +              fmha / fmoe / allreduce 热路径)
             AOT instance 库)                     ▲
                │                                 │
                ▼                            (与 CK 并列,只在 AITER 里出现)
            hipBLASLt
            (extended GEMM:
             FP8 + bias/activation
             fusion + grouped GEMM)
                │
                ▼
            rocBLAS  (经典 BLAS 基线,所有上层最终的落点之一)


                  ┌──────────────  legacy / portability  ──────────────┐
                  │                                                    │
       hipBLAS  (一份 public API, build-time 选 backend)
                  │                                                    │
                  ▼                                                    ▼
              rocBLAS (AMD)                                       cuBLAS (NVIDIA)

           注:hipBLAS 已退役,代码搬入 ROCm/rocm-libraries monorepo;
               现代训练 / 推理路径绕开 hipBLAS,直接吃 hipBLASLt 或 rocBLAS。
```

关键观察:

- 上半的两棵树(训练 / 推理)在 **fused-op 层显式分叉**(Primus-Turbo
  vs AITER),在 **kernel-template 层重新汇合**(都吃 CK / ck_tile /
  Triton)。
- 手写 ASM 这一档**只挂在 AITER 下**;Primus-Turbo 没有 `hsa/<arch>/*.co`
  的对应物,训练侧的极热路径靠 CK / hipBLASLt + 自有 `csrc/kernels/`
  覆盖,而不是 ASM。
- hipBLAS 是一条**独立的兼容性分支**,与左右两棵树无依赖关系;它的存在
  价值是 marshalling pattern 本身,不是性能。

## 2. 重复出现的设计模式 (recurring patterns)

| Pattern | 一句话:解决什么 | 体现在哪些库 |
|---|---|---|
| **Templated dispatch + per-(dtype, arch) instance enumeration**(AOT 把 (dtype × layout × tile × pipeline × arch) 笛卡尔积实例化成 `.cpp`/`.a`/`.so`) | "运行期不做 JIT、不抖动"的稳定 ABI,代价是编译时间和二进制体量 | [`composable-kernel.md`](composable-kernel.md) Layer 3(`library/` 下 ~1700 个 instance `.cpp` / 218 个 op-config 目录) |
| **Tile / block / warp policy 作为一等公民类型**(Pipeline = `Problem` × `Policy` × `Scheduler`,搬数据策略与算法分离) | 加新调度策略(double-buffer / async direct-to-LDS / weight preshuffle)不必复制 kernel 主体 | [`composable-kernel.md`](composable-kernel.md) `ck_tile/` 全套 |
| **Tensor coordinate transformation / layout abstraction**(ND tensor 的 padding / permute / merge / unmerge / xor swizzle / im2col 都做成 compile-time 可组合 transform) | "为每种 layout 重写一份 kernel" 的爆炸 | [`composable-kernel.md`](composable-kernel.md) 第二支柱 |
| **JIT compile + on-disk kernel cache + multi-process file-lock**(首调编译落 `~/.<lib>/build/<key>/*.so`,多进程 `FileBaton` lock) | 想覆盖 10+ dtype × 几十种 quant scheme × 多 arch,但不愿付 CK 那种 AOT 全枚举的安装包成本 | [`aiter.md`](aiter.md) `aiter/jit/core.py`(`@compile_ops` + `build_module` + `FileBaton`) |
| **Op registry + autograd.Function wrapper**(每个 op 是 Python 一等公民,要么 forward-only thin op、要么直接 `torch.autograd.Function`) | 框架直接 `import` 即可拿到 op,无需 graph-level 改造 | [`aiter.md`](aiter.md) 走 thin op + 框架自包 autograd;[`primus-turbo.md`](primus-turbo.md) 走 `torch.autograd.Function` 直接暴露 |
| **Marshalling layer for portable BLAS**(一份 public header,两个 backend `.cpp` 平行实现,build 时选一个) | 上层调用代码不想为 AMD/NVIDIA 写两份 | [`hipblas.md`](hipblas.md) 全库就是这个模式(`amd_detail/hipblas.cpp` + `nvidia_detail/hipblas.cpp`) |
| **Single-header HIP utility template library (Opus 风格)**(`opus.hpp` 单 header + `hip_minimal.hpp` 子集 + `--genco` + ctypes 入口,显式 build-time-first) | torch-extension 编译 8 s 是开发体验的最大杀手;不想退回纯 HIP 但又不能接受 CK 那种模板深度 | [`aiter.md`](aiter.md) `csrc/include/opus/`(对 `warp_sort_bitonic` 报 21 s → 346 ms 的 61× 编译加速) |
| **Multi-backend dispatcher inside a single op**(同一个 Python op 内部按 (dtype, shape, arch, env) 选 CK / Triton / hipBLASLt / 手写 ASM) | 框架方不该被要求知道 "什么 shape 走什么 backend",这层领域知识应当收编到库内 | [`aiter.md`](aiter.md)(`aiter.fused_moe` / `aiter.mha_fwd` 内部 dispatch);[`primus-turbo.md`](primus-turbo.md)(`BackendType` enum + `AutoKernelDispatcher`) |
| **"站在 vendor lib 之上" vs "自己写 kernel" 的取舍**(三条路:全自有 / 全外挂 / 混合) | 内部团队投入 vs 上游迭代速度的平衡 | 全自有:[`composable-kernel.md`](composable-kernel.md);全外挂(只 dispatch,几乎不写 kernel):[`primus-turbo.md`](primus-turbo.md)(自有只到 `csrc/kernels/` 几个轻量 fused + `TURBO` 兜底);混合(外挂 CK / Triton + 自有 ASM 热路径):[`aiter.md`](aiter.md) |
| **Tuned-config 入仓 + autotuning pipeline 入 CI**(每个 op 的 (shape → 最优 config) 以 CSV 形式版本化,CI 周期 sweep 更新) | tuning 结果只在工程师本地等于没有 | [`aiter.md`](aiter.md) `aiter/configs/*.csv` + `docs/autotuning_pipeline.md`;[`primus-turbo.md`](primus-turbo.md) 的 `TuneCache` 是运行期等价物 |
| **AOT instance + runtime best-of-N + `op_ptr` cache**(Client API 三件套:`MakeArgumentPointer` / `IsSupportedArgument` / `Run`,加 `InstanceFactory<>::GetInstances()`) | "稳定 ABI + 实测选 instance" 这两个看似冲突的需求同时满足 | [`composable-kernel.md`](composable-kernel.md) Layer 4;[`primus-turbo.md`](primus-turbo.md) `AutoKernelDispatcher` 是它的 Python 等价物 |
| **Explicit non-goal as design tool**(README 第一段就钉"不写 kernel、性能在后端" / "不是 framework / 不是图编译器") | 抽象层会被持续要求加 feature 直到自己变重 | [`hipblas.md`](hipblas.md)("not a performance library");[`aiter.md`](aiter.md)("不是图编译器");[`primus-turbo.md`](primus-turbo.md)("不是 framework,不带 dataloader / scheduler") |

## 3. 训练 vs 推理 分工 (training vs inference)

| 维度 | Primus-Turbo (训练) | AITER (推理) |
|---|---|---|
| 服务对象 | Primus-LM 经由 `primus/backends/megatron/...` 适配层,被 Megatron / TorchTitan 间接消费 | vLLM / SGLang / ATOM / JAX(README 自述 "default kernel backend for LLM inference on AMD GPUs") |
| op 表面 | `torch.autograd.Function` 直接暴露(`gemm_fp8.py` 里 `FP8GemmTensorFunction` / `FP8GemmRowFunction` 等)+ `nn.Module` 层(`Float8Linear`) | forward-only PyTorch op(`aiter.mha_fwd` / `aiter.fused_moe` / `aiter.paged_attn` / `aiter.mla`),autograd 由框架自己包 |
| 默认 dtype | FP8 HYBRID(E4M3 fwd + E5M2 bwd)、bf16、FP4;`Format` × `ScalingGranularity` 笛卡尔积 | bf16、FP8、FP4、INT4/8 quant 全家;按 production 模型(Kimi-K2.5 / DeepSeek-V3)反向调优 |
| 通信 | DeepEP / rocSHMEM(MoE EP all-to-all,**训练必需**;gfx942 only) | 部分 allreduce + RMSNorm fused(decode 路径) |
| AOT vs JIT | **AOT 倾向**:站在 CK 已编译 instance 与 hipBLASLt 之上,Triton 自身才走 JIT | **JIT 倾向**:`aiter/jit/` 首调编译落 `~/.aiter/jit/build/`,多 rank `FileBaton` lock,环境变量子集化 |
| 自有 kernel 占比 | 极小(`csrc/kernels/{quantization,normalization,moe_permute,shuffle,reduce}`),且只在 dispatcher 落空时走 `TURBO` 档 | 中等(`csrc/kernels/`)+ **重资产手写 ASM**(`hsa/gfx942/*.co` / `hsa/gfx950/*.co`,MLA decode / FMHA / fused MoE / allreduce 热路径) |
| 状态性 | **op 无状态**;FP8 weight cache 一律推给上层 framework adapter(`PrimusFP8GroupedMLP`,见 [`../kernels/fp8-expert-gemm.md`](../kernels/fp8-expert-gemm.md)) | tuned CSV(`aiter/configs/`)+ `.co` 二进制是库自身的版本化资产,生命周期与 release 绑 |
| 默认 dispatcher 优先级 | user setter > env var(per-precision,如 `FP8:HIPBLASLT,FP4:AITER`) > auto-tune(`TuneCache` LRU) > code default > fallback 遍历 | env var(`AITER_USE_CK_MOE_SORTING` 等) > tuned CSV 命中 > 库内 fallback |

灰区:**GEMM / RMSNorm / RoPE 这些两边都用得到的 op,两边各包一遍是显式
工程成本**,但比硬塞进一个库要清楚。我们自己规划新算子时先回答 "训练还
是推理" 再决定 op 集合与 backend 矩阵。

## 4. 算子开发选型建议 (decision table)

| 我想做什么 | 默认路径 | 为什么 |
|---|---|---|
| 新 fused GEMM on MI355X(训练) | 在 hipBLASLt(FP8 / grouped)或 CK ck_tile 上加 backend,落到 `primus-turbo` 的 `kernels/<op>_impl.py`,通过 `BackendType` 暴露 | "站在 vendor lib 之上"的薄壳模式;`TURBO` 档只在 dispatcher 落空时兜底 |
| 新 fused attention(推理) | 在 AITER `aiter/mha.py` / `aiter/paged_attn.py` 加 backend,优先复用 ck_tile FMHA + `hsa/<arch>/*.co` 手写 ASM | AITER 是 vLLM / SGLang ROCm 默认后端,新 op 在这里上线即同时进 production 推理栈 |
| 新 fused attention(训练) | 在 `primus-turbo` 加 backend,默认走 CK ck_tile 或 Triton;**不要复用 AITER 的 forward-only op**,因为缺 backward + 缺 autograd 包装 | 训练 op 必须是 autograd.Function;复用 AITER 会要求把整套 backward 在框架侧重写 |
| 跨 ROCm / CUDA 的可移植 BLAS 调用 | hipBLAS public API,但**使用 `ROCm/rocm-libraries` monorepo 版本**,不要直接拉 `3rd/hipBLAS/` | hipBLAS 自身仓库已退役(branch `develop_deprecated`);marshalling pattern 还在,但实现已经搬家 |
| 想 ship JIT'd kernel 进训练 stack(Triton / HIP) | 直接借用 AITER 的 cache 协议:`md_name` 包含 `(op, dtype, gfx, ROCm version, flag hash)`,落共享存储,多 rank 用 `FileBaton` lock | 多 rank 并发 + 多节点共享 cache 是已知的坑,AITER `aiter/jit/core.py` 已经踩过 |
| 小 HIP 工具 kernel(Pilot 自动 sweep / MonolithMoE 工具头)想要 < 1 s 编译 | Opus 模板(单 header HIP + `--genco` + ctypes 入口),而不是 torch extension | Opus 显式 build-time-first;对 `warp_sort_bitonic` 这种小 kernel 已实测 61× 加速(21 s → 346 ms) |
| 新 MoE EP all-to-all(训练) | `primus-turbo` `deep_ep/`(目前 gfx942 only,gfx950 等待),或自己扩展 | AITER 不覆盖训练 EP 通信;DeepEP / rocSHMEM 是 primus-turbo 一侧的专属 |
| 新 FP4 / MX FP8 GEMM(训练) | CK ck_tile Layer 1 warp dispatcher 加 MFMA 指令(`__builtin_amdgcn_mfma_scale_f32_*_f8f6f4`)+ Layer 3 加一组 instance + `primus-turbo` 加 `BackendType` 档 | 新 MFMA 指令的"干净扩展点"在 CK Layer 1;autograd 包装在 primus-turbo |
| per-(shape, dtype) best-of-N 选 backend | 直接用 `primus-turbo` 的 `AutoKernelDispatcher` + `TuneCache` LRU(graph capture 时跳过 tune) | profile-and-cache + LRU + graph-capture skip 三件事都写好了,不要现造 |
| 旧 cuBLAS 应用 1:1 迁移到 ROCm | hipBLAS(`rocm-libraries` 版),call-site 不改 | marshalling 模式的官方落点,唯一一份"接口跟 cuBLAS-v2 完全对齐"的库 |
| op 调度配置在多 env var 间散开 | 借 `primus-turbo` 的 `PRIMUS_TURBO_{GEMM,GROUPED_GEMM,...}_BACKEND=FP8:HIPBLASLT,FP4:AITER,OTHER:CK` 语义 | 一个变量 + per-precision 段,避免 `PRIMUS_FP8_USE_HIPBLASLT` / `PRIMUS_FP4_USE_AITER` 这种逐 op 变量蔓延 |

## 5. 我们最该借鉴的 3 个模式 (top 3 patterns to borrow, ranked)

### 5.1 "站在 vendor lib 之上" 的薄壳模式 + `AutoKernelDispatcher`(from primus-turbo + AITER)

这条排第一,是因为它直接决定**我们投入的每一个工程小时是叠加在 ROCm 团
队的输出上,还是平行重劳动**。primus-turbo 的整套 `BackendType` 枚举 +
`KernelBackend` ABC + `AutoKernelDispatcher` 把 "如何选 backend" 这件事
从散落在每个 op 里的 `if`-chain 收成一个中枢,加上 per-precision 环境
变量解析(`FP8:HIPBLASLT,FP4:AITER,OTHER:CK`)与 `TuneCache` LRU,几乎
就是我们想要的模板。AITER 的 op-internal multi-backend dispatch 是它的
姐妹形态,差别只是 "中枢一份" vs "每 op 一份"。

**最先该落地的地方:** `PrimusFP8GroupedMLP` 当前在 Python 里硬编一条
Triton 路径(见 [`../kernels/fp8-expert-gemm.md`](../kernels/fp8-expert-gemm.md)
的结论:280 ms 端到端差距里 277 ms 来自模块框架而非 GEMM)。把这条路径改
成 `BackendType` dispatch —— hipBLASLt grouped GEMM / CK ck_tile grouped
GEMM / 现 Triton 实现 收进同一个 `torch.autograd.Function`,内部按
`(E, tokens_per_expert 直方图, dtype, gfx)` dispatch ——
就同时拿到 "best backend 自动选" 与 "未来加新 backend 不动 call-site"
两个收益,而且不需要先证明哪条 backend 最快。

### 5.2 JIT cache 约定:`md_name` + `FileBaton` + 共享 cache dir(from AITER)

排第二是因为它是**唯一一个我们今天还没付的入门成本**。Pilot 自动 sweep
与 cco super-kernel 的中间 kernel 产物管理,迟早要面对"多 rank 并发
hipcc" / "多节点共享 cache" / "cache key 漏掉 hipcc flag 导致 silent
ABI mismatch"这三个坑;AITER `aiter/jit/core.py` 已经把答案完整写完,
连环境变量(`AITER_REBUILD` / `AITER_JIT_DIR` / `ENABLE_CK` / `GPU_ARCHS`)
都给出了 reference design。

**最先该落地的地方:** Pilot 自动 sweep 产出的 Triton / HIP 中间 kernel,
按 `(op, dtype, tile shape, gfx, ROCm version, hipcc flag hash)` 落到
SLURM 共享存储 + `FileBaton` lock。同样的协议直接复用到 MonolithMoE
super-kernel 的 codegen 产物。代价是一次性写 ~150 行 Python,收益是后续
所有 JIT'd kernel 不必再回答 "怎么避免 step-0 编译撕开" 这个问题。

### 5.3 Tensor coordinate transformation + tile/policy as first-class type(from CK)

排第三但**最深**。前两条是工程基础设施,这一条是**我们自己写 kernel 时
的天花板**。CK 把 layout(padding / permute / merge / xor swizzle / im2col)
全部做成 compile-time 可组合 transform,把"搬数据策略"(`Policy`)与
"算什么"(`Problem`)分成独立类型 —— 这是 "为每种 layout 重写一份
kernel" 与 "为每种调度重写一份 kernel" 这两类爆炸的根因消除剂。

**最先该落地的地方:** MonolithMoE super-kernel 的 inner-loop 当前是单一
HIP 模板,加新调度策略(async direct-to-LDS / weight preshuffle / 不同
software pipeline 阶段数)就要复制整支 kernel。下次重构时按 CK 的
"`Problem` × `Policy` × `Scheduler`"拆,kernel 主体只认 tile,policy
作为独立 `constexpr` 类型 / Triton `tl.constexpr` —— review 时硬挡住把
policy 知识塞回 kernel 主体的 PR。同样的拆法也适用于 FP8 grouped GEMM:
`fp8_grouped_gemm.py` / `bf16_fused_grouped_gemm.py` 把 tile shape 与
pipeline policy 拆成独立类型,之后加 gfx950 的 `mfma_scale_f32_*_f8f6f4`
就只动 warp dispatcher 表,不动 kernel 模板。

## 6. TODO / 待补充

四份单库蒸馏完成后,下面这些是被反复引用但还没有专文的对象,**任何一项
被蒸馏后,本文 sections 1–5 都需要回头修一次**:

- **hipBLASLt** —— 承接 hipBLAS 的现代化层,是 primus-turbo / CK /
  AITER 三家共同消费的 GEMM 中枢(FP8 input、bias+activation fusion、
  grouped GEMM 全在它身上)。目前在 `hipblas.md` 和本文里都只是被引用,
  需要单独一篇 `hipblaslt.md`。
- **rocBLAS** —— 本谱系图的真实底层,hipBLAS / hipBLASLt 的最终落点之
  一。蒸馏后能讲清 "Tensile 模板路线 vs CK ck_tile 路线" 在 AMD 内部
  的分工。
- **Triton on ROCm** —— 上面四个库里有三个都把 Triton 当成 backend
  之一,但 Triton 自身在 AMD 上的优化路径(matrix-core lowering / LDS
  swizzle / shared-memory layout)还没沉淀。
- **DeepEP / rocSHMEM** —— primus-turbo 训练 EP 通信的实现,目前只在
  `primus-turbo.md` 里被点名,gfx950 可用性也待 `notes/` 跟踪。
- **实测:cross-library dispatch 冲突** —— `primus-turbo` 的
  `PRIMUS_TURBO_*_BACKEND` 与 Primus-LM 框架层的
  `_use_fused_fp8_triton` / `_use_hipblaslt_fp8` 等环境变量在真实训练
  step 里如何相互覆盖,目前是猜测,需要一次实测后归 `notes/` 再回填
  到本文 Section 4 决策表。
- **跨库 best-backend 分布实测** —— 在 DSv3 / Kimi-K2.5 这种 production
  workload 上 `AutoKernelDispatcher` 真实命中的 backend(FP8 GEMM 多
  hipBLASLt 还是多 CK?GroupedGEMM 实际几个走 Triton?),决定本文
  Section 4 默认推荐的精确度。
- 蒸馏满 6 个库之后(加上 FlashAttention / CUTLASS / TransformerEngine
  这三个 NVIDIA 侧对照物中的至少一个),本文要扩出一个 "AMD 栈 vs
  NVIDIA 栈" 的横切对比段落,而不是现在这个纯 ROCm 视角。
