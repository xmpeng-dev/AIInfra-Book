# Ave / avelang (causalflow-ai)

> **Repo:** `causalflow-ai/avelang` &nbsp; **Local path:** 未纳入 `3rd/` — 临时 clone `/tmp/avelang`
> **Snapshot:** `cf42479` &nbsp; `2026-07-12` &nbsp; on branch `master`
> **Size:** ~2.6 MB &nbsp; **License:** Apache-2.0
> **Distilled:** 2026-08-04
> 来源:README + `docs/content/`(installation / tutorials / language-reference)
> + `lib/Driver/driver.h`、`lib/Dialect/AveLang/IR/AveLangOps.td`、
> `lib/IR/Intrinsics/*.mlir` 顶部 + `python/avelang_kernels/amdgpu_gemm.py`。
> Docs: <https://www.causalflow.ai/docs/avelang>。

## TL;DR

Ave 是 causalflow.ai 的**实验性 Pythonic GPU kernel 语言**:Python 语法当
前端,自建 MLIR pipeline 进程内直出 hsaco/cubin。它的设计赌注是
**Triton 的反面**——不做 layout 推导、不做自动软件流水、不做 autotune,把
执行几何、LDS 布局、MFMA fragment 打包**乃至 `s_waitcnt` /
`sched_group_barrier` / `s_setprio` 全部提升为一等 Python 语法**。

对我们真正有价值的只有一点:它的 **intrinsic = 内嵌 MLIR 文本库 +
`always_inline` wrapper** 模式,是目前见过给 gfx950 补新指令最便宜的扩展
路径(3.4 / §4 第二行)。

**先破除一个误读**(本文初稿犯过):"它把 `s_waitcnt` 做成一等语法" **不
构成对我们的能力增益**。HIP 早有 `__builtin_amdgcn_s_waitcnt` /
`sched_group_barrier` / `s_setprio`,我们的 `grouped_gemm.hip` 本来就在
用;两条路落到同一个 LLVM AMDGPU 后端、同一套 RA 与同一个
`SIInsertWaitcnts`,**代码质量上限相同**。相对手写 HIP,avelang 的实际
差异只在开发体验(免 hipcc 往返、typed layout/view 抽象),不在能力。

**不是可落地的库**:alpha(21 commits / 3 作者 / 最后提交 2026-07-12)、
构建要自备 LLVM+MLIR+Clang **22**、只有进程内 kernel cache、仓库无 CI、
默认 target 还是 **gfx90a(MI200)**。当**设计参考**读,与 `mkernel.md`
同类。

## 1. 库定位 (positioning)

- **一句话:** 一门用 Python 语法书写、经自建 MLIR pipeline 编译的
  **零抽象 GPU kernel DSL**,目标是"让人(和 LLM)在没有隐藏语义的语言里
  写出 CK/汇编级 kernel"。
- **它是什么:**
  - 一条**完整自建的编译链**:Python `ast` → 编译器内部 AST(C++ 数据
    结构,**不是生成 C++ 源码**)→ 自有 MLIR dialect `ave` →
    memref/scf/arith → GPU dialect outlining → ROCDL / NVVM → LLVM IR →
    GPU code object。C++ 侧约 26.9K 行,是仓库主体。
  - **不经 hipcc / 不编译 C++ 源码**:JIT 时前端到 LLVM IR 全在进程内内
    存中完成,省掉的正是 hipcc 那趟 clang C++ 前端 + 模板实例化(这部分
    有多贵,可参考 [`aiter.md`](aiter.md) §3.4 记的 CK 模板 → Opus 轻模板
    21 s → 346 ms —— 那仍在 hipcc 内,只作开销量级的旁证)。注意最后一步
    device link 仍会 `ExecuteAndWait` 起子进程调 `ld.lld`/clang 链 ROCm
    device library bitcode(`lib/Target/AMDGPU/amdgpu_backend.cc`),这一
    步所有 AMD 工具链都躲不掉,但链的是 bitcode 而非 C++。
  - 一套 **CuTe 风格的极简 tensor/layout 语义**:`al.Tensor(shape,
    stride, dtype)`、`al.make_layout`、`al.view` / `al.subview` 零拷贝
    重解释,layout 直接进 MLIR 类型
    (`!ave.memref<!ave.layout<dims,strides>, T>`)。
  - 一层**裸 intrinsic 表面**:AMD 侧约 30 个(MFMA、raw-buffer
    load/store、DPP、`cvt_pk_fp8/bf8`、atomic add、调度类),NVVM 侧
    MMA + `ldmatrix`/`stmatrix`。
  - 一个**从 Triton 逐行移植的 runtime**(见 3.6)。
- **它不是什么 / 不做什么:**
  - **不是 Triton 竞品**:不做 layout 推导、不做自动 coalescing /
    软件流水、没有 autotune 装饰器。launch 时 grid **和** block 都要你
    显式给(`fn[lambda: (grid, block)](args)`),没有 `num_warps` 抽象。
  - **不是算子库**:全仓库只有一个示例 kernel(`avelang_kernels/
    amdgpu_gemm.py`,BF16 GEMM),没有 op registry、没有 dispatcher。
  - **不是 production 依赖**:官方自述 alpha,"APIs, generated code, and
    backend coverage may change"。
- **谁在用:** 只有 causalflow 自己(6 stars / 3 forks / 7 open PRs)。
  三个作者,2026-05-07 起,最后一次提交距蒸馏日约 3 周。

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| `lib/Frontend/` | 两条前端入口:文本 `.ave` 源 parser,以及 **把 Python AST 对象跨 pybind11 直接吃进来** 的 `ParsePythonFunctionDef` | JIT 路径走后者,没有自研 lexer/grammar |
| `lib/AST/` + `lib/Basic/` | 自有 clang 风格 AST;诊断复用**真正的 `clang::DiagnosticsEngine` + `SourceManager`** | 诊断码目前只有 8 个(`kUnimplemented` / `kTypeMismatch` …) |
| `lib/Dialect/AveLang/` | 自有 dialect `ave`:layout/tuple/full/return + 后端 intrinsic op;配套 4 个自定义 pass(hoist-alloca、return 归一化等) | layout 是**类型的一部分**,不是 attribute |
| `lib/IR/` | AST → MLIR 的 generator(type promotion / symbol scope / constexpr 注入 / mangler) | `constexpr` 以 JSON 从 Python 侧注入 |
| `lib/IR/Intrinsics/` | **intrinsic 库以 MLIR 文本形式内嵌**(`amdgpu_intrinsics.mlir` / `nvvm_intrinsics.mlir`),按需 link | 本库最值得抄的一处,见 3.4 |
| `lib/Target/{GPU,AMDGPU,NVVM}/` | 后端注册表 + 各自的 lowering pipeline(大量复用上游 MLIR conversion pass)+ ROCm 安装探测 | AMDGPU pipeline 里 wave64 由 target feature 决定 |
| `python/avelang/{runtime,compiler,backends}/` | Triton 移植的 JIT / 特化 / 缓存 / launcher,以及 CUDA/HIP driver glue | 约 4.4K 行 |
| `python/avelang_kernels/` + `benchmark/` + `test/` | 唯一的示例 kernel(MI300 级 BF16 GEMM)+ 其 CUDA-Graph benchmark + 24 个 pytest | AMD 侧 7 个测试文件,NVIDIA 侧 2 个 |

## 3. 核心设计理念 (core design ideas)

### 3.1 "显式优于推导" —— 抽象层次刻意压到 CK 之下

文档原话:"explicit control over **execution geometry, memory layouts,
and backend intrinsics**"。这是字面意思:你写 `al.thread_id(0)`、自己
把 linear tid 拆成 lane→row/col、自己 `al.make_shared` 分配 LDS、自己
把 fragment 打包成 per-lane 的 packed `u32` 再喂给 MFMA。

| | 谁决定 layout / 软件流水 | 你写什么 |
|---|---|---|
| Triton | 编译器 | block 级程序,`tl.load` / `tl.dot` |
| **Ave** | **你** | thread/lane 级索引 + 显式 LDS + 逐 lane MFMA fragment |
| CK / ck_tile | 你(C++ 模板元编程) | `Problem` × `Policy` × `Scheduler` |

它换来的不是生产力,而是**没有任何隐藏语义**:生成的指令序列基本可由
源码逐行预测。代价见 4 的最后一行。

### 3.2 Python 就是语法,诊断却是 clang 级

`@avelang.jit` 取源码 → `ast.parse` → 把 **Python AST 对象**送进 C++,
在 C++ 侧翻译成自有 AST。因此没有自研 grammar 要维护,语言表面天然
"是合法 Python"。关键工程细节是它用 `ast.increment_lineno` + 一段
padded source buffer 喂给 clang 的 `SourceManager`,**使报错能精确指回
用户 `.py` 的行列**。对一个 DSL 来说这是罕见的完成度,也是它敢说自己
适合 agent 迭代的唯一实质依据(见 3.7)。

但 parse 模型有个容易被忽略的后果:**函数体不执行,所以借了 Python 的语
法却借不到 Python 的元编程**。全仓没有 `static_for`、没有 unroll pragma、
前端 parser 里也没有任何 constexpr 求值——唯一的展开是
`lib/Target/GPU/lower_to_llvm.cc:282` 打开 LLVM 自己的 `LoopUnrolling`。
它的示例 GEMM 就是这么写的:tile 参数是模块级 Python 常量
(`M_TILES_PER_WARP = WARP_MAT_M // 16`),循环写 `al.range()` 生成真
loop,再指望 LLVM 展开。C++ 侧 `template<int N>` + `#pragma unroll` 那种
编译期代码生成,这里做不到;FlyDSL 因为 Python 真跑,反而能用
comprehension/闭包生成 IR。

**因此"avelang vs 手写 HIP"的净收益只剩一条:绕开 hipcc C++ 前端换来的
编译速度。** 编程模型(SIMT / 显式 LDS / 显式寄存器)一样,后端一样,元
编程能力更弱,库生态为空(HIP 那边可 `#include` CK / rocPRIM / hipCUB),
且不生成 debug info(`lib/` 内 `DebugInfo`/`DILocation`/dwarf 命中 0 处,
rocgdb 链路用不上)。唯一超出语法层的真差异是 3.3 的 layout 进类型系统。

### 3.3 layout 进类型系统,view/subview 是纯重解释

tensor 类型带 shape + 可选 stride,坐标按 `Σ cᵢ·tᵢ` 线性化;嵌套 shape
用来表达 packed/硬件布局(例如把连续 BF16 行看成
`(row, vector, word)`)。`al.view` / `al.subview` **不搬数据**,只换逻辑
视图——同一块 LDS 既能当 `bf16` 又能当打包 `u32` 向量,这正是喂 MFMA
fragment 和做向量化访存的机制。这是 CuTe/CK "tensor coordinate
transformation" 的极简版:只有 shape/stride/nesting,没有 CK 那套
merge/unmerge/xor-swizzle transform 代数。

### 3.4 intrinsic = 内嵌的 MLIR 文本库 + `always_inline` wrapper

`lib/IR/Intrinsics/amdgpu_intrinsics.mlir` 是一段**嵌进二进制的 MLIR
源码**,每个 intrinsic 就是一个标了 `func.inline = "always"` 的
`func.func`,body 里包一条 `rocdl.*` 或 `llvm.call_intrinsic`,并在里面
顺手做好 `vector.bitcast` 这类形状适配。文件顶部注释点明了取舍:

> Keep this library textual source as the authority; LLVM bytecode is
> version-sensitive and must be regenerated by the current mlir-opt.

于是**加一条新硬件指令 = 加一个 `func.func` + 一个 Python 空壳函数 +
注册名字**,不需要 TableGen op + conversion pattern + verifier 三件套。
效果直接体现在提交历史上:最近 11 个 commit 里有 9 个是
`[IR][amdgpu] Support <指令>`(DPP、`s_setprio`、`sched_barrier`、
`v_setvskip`、`readfirstlane`、`cvt_pk_fp8/bf8`、FP8 MFMA、atomic add)。

### 3.5 调度控制本身也是语言表面

这是 Ave 与**其他 kernel DSL**(Triton / CuTe DSL / ThunderKittens)最不
同的一点:**后端调度器 hint 不是逃生舱,而是普通 builtin**。
`al.amdgpu.s_waitcnt(vmcnt, expcnt, lgkmcnt)`(三段分开写,各按硬件范围
做编译期校验)、`sched_group_barrier(mask, size, group_id)`、
`sched_barrier(mask)`、`s_setprio`、`v_setvskip`、`readfirstlane` 全部直
接可调。示例 kernel 里甚至把一整段调度脚本单独抽成一个 `@avelang.jit`
函数复用。

但**对 HIP 不构成差异**:这些 builtin 在 HIP 里一个不少
(`__builtin_amdgcn_*`)。差别只是 avelang 把 packed immediate 拆成三个
带范围校验的参数。语言层的"显式调度"叙事在与 Triton 对比时成立,在与
手写 HIP 对比时不成立。

### 3.6 runtime 不自研:直接移植 Triton 的 launch path

`python/avelang/runtime/jit.py` 是 Triton runtime 的近乎逐行移植:
`DependenciesFinder`、`used_global_vals` 一致性校验、用 `exec` 生成
binder 的 `create_function_from_signature`、`specialize_impl`、
per-device kernel cache、`knobs.py` env-backed settings——连 Triton 的
注释和一个 `TODO(jlebar)` 都还在。这反过来证明一件对我们有用的事:
**Triton 里真正难且值钱的是 launch-path 特化 + cache-key 机制,它可以
和编译器解耦**。注意它只抄了一半:cache key 计算(`ASTSource.hash()`)
在,**落盘的 cache manager 没抄**,`compile()` 直接调
`backend.compile()`,所以每起一个新进程都要重付一次完整 MLIR+LLVM 编译。

### 3.7 "designed for LLM agents" 目前是定位,不是机制

docs 首页原话:"By combining tile-based access models with **program
analysis**, Ave provides concrete guidances for LLMs to generate
real-world performant GPU kernels."。实际查证:仓库里**没有**
`AGENTS.md`、没有 rules、没有 MCP server、没有结构化错误目录(诊断码
共 8 个)、没有 cost model、没有那个被承诺的 analysis pass。

真正可能帮到 agent 的都是**间接效应**,但也确实成立:Python 语法对 LLM
是 in-distribution;语言表面很小(约 40 builtin + 30 intrinsic)对比
CK/CUTLASS 的巨型模板表面;3.2 的 clang 级诊断 + 正确行号是 agent
迭代循环的好反馈信号;`tools/dump_llvm_ir.py` / `dump_assembly.py` 让
agent 能检查自己的产物。**设计意图自洽,机制还没写。**

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| **调度 / 等待语义纳入类型化语言表面**(`s_waitcnt(vmcnt,expcnt,lgkmcnt)` 三段分开写且编译期校验硬件范围,而非 HIP 里一个打包好的 magic immediate;`sched_group_barrier` / `sched_barrier` / `s_setprio` 同理) | 手算打包 waitcnt immediate、或记不住 `vmcnt` 上限是 63 还是 15,是易错点 | **可读性/防错收益,不是能力收益**。若我们要在 MonolithMoE 里长期维护手写 waitcnt,值得照它的样子包一层带范围断言的 inline wrapper | **这一行的价值很薄,不要读高**:HIP 早已有 `__builtin_amdgcn_s_waitcnt` / `sched_group_barrier` / `s_setprio`,[`grouped_gemm.hip`](../../notes/monolith-moe/2026-05-12_1955_sched_group_barrier_p0_failed_compiler_waitcnt_pass_forces_lgkmcnt0.md) 本来就在用。avelang 与 HIP 落到同一个 LLVM AMDGPU 后端、同一套 RA 与同一个 `SIInsertWaitcnts`,**代码质量上限相同**,故它对那个 `lgkmcnt(0)` 根因零贡献 |
| **intrinsic 作为内嵌 MLIR 文本库**(`func.func` + `always_inline` 包 `rocdl.*`,`vector.bitcast` 形状适配写在 wrapper 里) | 给一个编译器补新硬件指令的成本从"TableGen op + conversion pattern + verifier"降到"一个函数 + 一个名字" | 给 gfx950 (MI355X/CDNA4) 补 `v_mfma_f32_16x16x32_bf16`(双速率)、scaled MFMA(`f8f6f4`)、`buffer_load_lds_b32` 这类指令时的扩展模式。对照 `_patterns.md` §6 里 "Triton on ROCm" 那条 TODO:往 Triton-on-ROCm 加一条 matrix-core 指令的成本比这个高一个数量级 | 便宜的代价是**没有 verifier、没有 layout 校验**:per-lane fragment 形状对不对全靠调用方。作为库的扩展机制很好,作为**用户面 API** 会把错误推迟到运行期出错数 |
| **Python AST 跨 pybind11 进 C++ 编译器,并用 clang `SourceManager` 保住 `.py` 行列** | 自研 DSL/codegen 的报错通常指向生成物而非用户源码,agent 与人都难迭代 | **Primus Pilot v2**(`knowledge/pilot/`)的 codegen/校验环节,以及任何我们自己写的 kernel 生成器 | **要借鉴请优先看 FlyDSL 而非本库**:`3rd/flydsl` 的 `python/flydsl/compiler/diagnostics.py` 用 `location_chain()` 把 MLIR location 走回 Python 调用栈(跳过合成 `<...>` 源、带代码片段、装 excepthook),用户可见效果同级、处理的是调用链、且**不把 clang/LLVM 版本耦合进来**——Ave 为这套诊断钉死了 LLVM 22 |
| **Triton runtime 与编译器可解耦**(照搬 `DependenciesFinder` + `exec` 生成 binder + specialization,换掉后端) | 想要 Triton 级的低 launch 开销,但不想用 Triton 的编译器 | 若将来给 MonolithMoE / RocMoE 做自有 kernel 前端,launch path 不必重写;也解释了为什么我们评估任何"新 DSL"时应把**编译器**和**runtime**分开打分 | Ave 恰好漏抄了落盘 cache(见 3.6)。抄的时候**必须连 on-disk cache + 多进程 file lock 一起抄**,否则训练场景每 rank 每次起进程都全量重编——参考 [`aiter.md`](aiter.md) 的 `FileBaton` 模式 |
| **示例 kernel 里的 MI300 手工常数,当作可读的 CK 平替读物**(XCC 感知 workgroup swizzle:`MI300_CU_COUNT = 38*8` / `WGM_XCC = 8`;LDS pad `SHM_PAD_BF16 = 16`;`sched_group_barrier` 的分组配比) | CK V3 的同类逻辑埋在模板里难读,而这里是 494 行平铺的 Python 常量 | 两处直接对照:(a) LDS pad 取 16 个 bf16 = 32 B,与我们 [`2026-05-12_2230_lds_pad_8_unlocks_25pct_grouped_gemm.md`](../../notes/monolith-moe/2026-05-12_2230_lds_pad_8_unlocks_25pct_grouped_gemm.md) 得到的 **`PAD % 8 == 0` 是 alignment 阈值** 相互印证;(b) 它的 XCC swizzle 可作 RocMoE persistent kernel group-mapping 的参考实现 | 常数是**gfx942 (MI300X/CDNA3) 假设**(`38*8` 是 MI300 的 CU 布局),gfx950 的 XCD/CU 数不同,不能直接搬;且该 kernel **无实测数据入仓**(benchmark 脚本在,结果没有),别把它当性能结论 |
| **反向教材:"零抽象 DSL" 的代价**(显式非目标:不做 layout 推导 / 不做流水 / 不做 autotune) | 抽象层会被持续要求加 feature 直到自己变重——Ave 走到了另一个极端 | 我们评估 kernel DSL 路线时的对照点:它的示例 GEMM 复杂度与 CK 版本**基本相同**,只是换成 Python 语法。即"换语法不降复杂度";真正省人力的是 CK 的 policy 复用或 Triton 的编译器,不是语法 | 这条是判断而非可执行模式;若哪天 Ave 真做出了它承诺的 program analysis,这条结论要回头修 |

## 5. 与生态的关系 (ecosystem position)

```
Ave / avelang
  ├── frontend  → Python `ast`(无自研 grammar)+ clang Diagnostic/SourceManager
  ├── middle    → 自有 MLIR dialect `ave` → 上游 MLIR(memref/scf/arith/gpu/vector)
  ├── backend   → ROCDL(gfx90a 默认,gfx942 部分) | NVVM(sm_80 / sm_90a)
  ├── intrinsic → 内嵌 .mlir 文本库(rocdl.* / llvm.call_intrinsic)
  └── runtime   → Triton runtime 移植(launch 特化 + 进程内 cache)
      依赖:LLVM + MLIR + Clang 22(自备,推荐 ROCm 7.2.x 的 LLVM fork)
```

Ave 坐在"**kernel DSL / 语言与编译器**"这一格,与 `knowledge/libraries/`
里已蒸馏的算子库**都不重叠**:CK / AITER / Primus-Turbo / hipBLAS 是
AMD 单卡算子库,`mkernel.md` 是 NVIDIA 多卡融合编排参考,而 Ave 是一门
**语言**。横向对标应放在 Triton / CuTe(CUTLASS Python DSL)/
ThunderKittens / Mojo 这一排里:Triton 把 layout 与流水交给编译器,CuTe
交给 C++ 模板,ThunderKittens 交给 tile 类型库,**Ave 全部交回给人**。

**最直接的对照物是 `3rd/flydsl`(ROCm/FlyDSL),而且它全面压制本库。** 同为
Python + 自建 MLIR dialect + 显式 layout + AMD intrinsic,但 FlyDSL 有
**带代数的一等 layout IR**(composition/product/divide/partition + copy/MMA
atom)、autotune、落盘 JIT cache、CI + 性能看板、PyPI 包、5.6 万行生产
kernel(含 4199 行 `mega_moe`),gfx950 是其主目标(仓库内 170 处引用 vs
本库 1 处)。**AMD 能力面上 Ave 基本是 FlyDSL 的真子集**,实测只有
`s_setprio` 与 `v_setvskip` 两个 intrinsic 是 FlyDSL 没有的。Ave 仅剩两点
非子集差异:(a) **NVVM 后端**(sm_80/sm_90a),FlyDSL 纯 ROCDL;(b) 前端是
**parse**(Python AST → C++ AST,语义在编译器里)而 FlyDSL 是 **trace**
(AST rewrite + Python 执行期 tracing,元编程更自由但有 tracing 陷阱)。
结论:AMD 侧 kernel DSL 选型应以 FlyDSL 为默认,本库只作语言设计参考。

它与 `_patterns.md` §6 的 "Triton on ROCm" TODO 是同一个问题的两个答案,
因此**没有回填 `_patterns.md` sections 1–5**:那五节的骨架是"vendor lib
之上叠 dispatcher"的 ROCm 算子栈谱系,Ave 不在那棵树上(与 `mkernel.md`
同处理)。等到我们真正要在"自研 DSL vs 扩 Triton vs 写 CK"之间做选型
时,再在 §4 决策表里加一行"语言层"。

## 6. 进一步阅读 / TODO

入口文件(≤ 5,各一句"为什么读"):

- `docs/content/language-reference/hardware-intrinsics.md` —— 一页看完
  AMD/NVVM 全部 intrinsic 表面与调用约定,判断"我们要的指令有没有"最快。
- `lib/IR/Intrinsics/amdgpu_intrinsics.mlir` —— 3.4 那个扩展模式的全部
  实现,不到 60 行,想抄看这个就够。
- `python/avelang_kernels/amdgpu_gemm.py` —— 唯一的真实 kernel;XCC
  swizzle / LDS pad / 双缓冲流水 / `sched_group_barrier` 全在里面。
- `python/avelang/runtime/jit.py` —— 对照 Triton 看"哪些抄了、哪些没抄"
  (尤其是缺失的 on-disk cache)。
- `lib/Target/AMDGPU/gpu_to_amdgpu_pipeline.cc` —— AMDGPU lowering 的
  pass 清单,看它自研了什么、复用了上游什么。

待验证 / 待沉淀(**不在本文,做了归别处**):

- **已排除,不要重跑**:"显式发 partial waitcnt" 这条路在
  [`1955 笔记`](../../notes/monolith-moe/2026-05-12_1955_sched_group_barrier_p0_failed_compiler_waitcnt_pass_forces_lgkmcnt0.md)
  阶段 3 已有结论——baseline 本来就带显式 `s_waitcnt lgkmcnt(8)`,编译器
  照样在其上再插 `lgkmcnt(0)`;删掉它也一样 ~180 T。**显式 wait 只能让
  等待更强,不能让它更弱**,换成 avelang 的 builtin 同理(同一个 intrinsic)。
- **仍未验证的两条**(都与 avelang 无关,归 `notes/monolith-moe/`):
  (a) 该笔记的 **P0a**——32 个 ds_read 全 issue 到不冲突寄存器再连做 64
  个 mfma,目标不是更细的 waitcnt,而是让 `lgkmcnt(0)` 每 K-tile 只付一
  次而非每 mfma 付一次;(b) **真正的 inline-asm block**(把 ds_read 与
  mfma 一起塞进同一 asm block 并绑寄存器约束,使 `SIInsertWaitcnts` 看
  不到寄存器依赖)——这与"调用一个 waitcnt builtin"是两件不同的事。
- Ave 的 tensor/layout 语义(只有 shape/stride/nesting)相比 CK 的
  transform 代数,在表达 gfx950 LDS XOR swizzle 时够不够用——与
  `knowledge/kernels/` 里 LDS swizzle 待独立成篇那条 TODO 合并考虑。
- 如果之后要把它纳入 `3rd/`:先解决 LLVM 22 自备构建 + 无 on-disk cache
  两个阻塞项,并确认 gfx950 支持度(全仓库 `gfx950` 只出现 1 次)。
- 与 Triton-on-ROCm / CuTe DSL / ThunderKittens 的语言层横向对比,若要
  做,单独走 `_patterns.md` §4 决策表扩行,不塞进本文。
