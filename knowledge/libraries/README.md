# knowledge/libraries/ — 第三方算子库领域知识

每个文件蒸馏 **一个** 第三方算子 / kernel 库的设计思路,作为后续算子开发
和库选型时的快速参考。生成流程见
`.cursor/skills/distill-operator-repo/SKILL.md`。

注意区分:

- `knowledge/kernels/` — 一篇文档 = 一个 **可移植 kernel 模式**(例如
  `fp8-expert-gemm.md`,与具体库无关)。
- `knowledge/libraries/`(本目录)— 一篇文档 = 一个 **第三方库的设计走读**。

## 索引

| 库 | 位置 | 角色 | 状态 |
|---|---|---|---|
| [Composable Kernel](composable-kernel.md) | `3rd/composable_kernel/` | AMD 通用 ML kernel 模板库(tile-based + tensor coordinate transform 两根支柱,AOT 枚举 instance,AMD 版 CUTLASS) | active |
| [AITER](aiter.md) | `3rd/aiter/` | ROCm production-grade op registry,JIT + on-disk cache + 同 op 多后端 dispatch(CK / Triton / ASM / Opus),vLLM/SGLang 默认后端 | active |
| [Primus-Turbo](primus-turbo.md) | `3rd/turbo/` | AMD 训练侧 fused-op 库,薄壳 dispatcher 站在 CK + hipBLASLt + AITER + Triton 之上,PyTorch/JAX 双前端 | active |
| [hipBLAS](hipblas.md) | `3rd/hipBLAS/` | 经典 BLAS marshalling 层(同一份 `hipblas.h` + 两个后端 `.cpp`:rocBLAS / cuBLAS thin wrapper);现代 LLM 训练实际消费的是 sibling `hipBLASLt` | deprecated → rocm-libraries |
| [mKernel](mkernel.md) | github `uccl-project/mKernel` | NVIDIA-only(sm_90a)多-node fused megakernel 算子集:persistent kernel + CTA 角色专精 + GPU-driven libibverbs 网络 + tile 级 comm/compute overlap;**设计参考**(对标 RocMoE/MonolithMoE super-kernel),非可落地 AMD 库 | reference |

## 跨库综述

- [`_patterns.md`](_patterns.md) —— 跨四份单库蒸馏的横切综述:库谱系
  图(训练 / 推理两棵树共享底层 kernel lib)、12 条重复出现的设计模式
  及其归属、训练 vs 推理分工对照、未来新算子的选型决策表、ranked
  top-3 应当借鉴的模式。新加单库蒸馏时回头修一次本文。

## 编辑约定

- 每个 `<slug>.md` 由 `distill-operator-repo` skill 产出,6 节
  (positioning / conceptual layout / 核心设计理念 / 可借鉴的设计模式 /
  与生态的关系 / 进一步阅读)。第 4 节 "可借鉴的设计模式" 是全文重
  心,条目必须点到具体 Primus 模块 / kernel / 工作流,否则视为不合
  格。约 150–300 行,只谈 idea,不做 file walkthrough、不做 dtype
  matrix、不放超过 3 行的代码片段。
- snapshot header 必须带 commit short-sha + 日期 + branch,这样
  note 老化时一眼能看出。
- 新增库时,同步在本表加一行,并在 `knowledge/README.md` 的子领域
  表里指向本目录。
