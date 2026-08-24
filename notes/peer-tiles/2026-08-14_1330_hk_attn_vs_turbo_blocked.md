# attention 全后端横测：HK vs turbo 的 aiter / Triton / fp8 / FlyDSL

> **When**: 2026-08-14 13:30 UTC+8（**14:05 定位根因 + 拿到 HK 数据，主结论从"blocked"改为"已完成"**）
> **Where**: `smci355-ccs-aus-n04-21`（8× MI355X / gfx950），容器 `xiaoming-dev`（`rocm/primus:v26.3`，ROCm 7.2.1 / hipcc clang 22 / torch 2.10.0+git94c6e04），单卡 `HIP_VISIBLE_DEVICES=0`
> **Context**: 按同形状对比 HK attention 与 Primus-Turbo attention 的性能

## TL;DR

1. **根因已定位，是 HK 的写法 × clang concept 判定，不是 ROCm 版本问题。** HK 的 attention kernel 用**别名模板** `attn_tile<D, T, L, S>` 声明 tile，而该别名展开里 **`D` 根本没被使用**（`= rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>`）。在模板内以依赖的 `D` 书写此类型时，clang 22 判不出三个 concept 约束（`ducks::rt::all` / `ducks::art::all` / `ducks::rv::all`）互斥，**三个重载同时可行 → 歧义**。同一类型改成非依赖写法即 0 error。
2. **修法是语义等价的最小改动**：pybind 只绑 `dispatch_micro<ATTN_D>`，kernel 实际只以 `ATTN_D` 实例化一次 → 把模板参数 `D` 固定成 `ATTN_D`（去模板化）。改完 `gqa` / `gqa_causal` 均编过（VGPR 238 / AGPR 0 / 2 waves/SIMD / 无 spill）。
3. **HK / 最快 turbo = 1.04–1.16×，几何平均 1.10×，6/6 全胜**（同进程背靠背，cos ≥ 0.999994）。与论文"attention forward 1.0–2.1× AITER"的**低端**吻合——合理，因为 MI355X 上的 aiter 已远强于论文的基线。
4. **turbo 三条路径排序稳定**，且"最快 turbo"在全部 6 行都是 **aiter CK/asm**（生产默认）：**CK/asm > aiter Triton 0.59–0.80× > Triton fp8 0.19–0.47×**。fp8 慢 2.4–5×，与 DSV3 报告的「慢 4.0×、精度低 25 dB」一致。
5. **FlyDSL 那条不可与上表并列**——它在 turbo 里只有 DSV4 **sparse MLA** 一个算子（`kv_lora_rank=512`/`d_qk=576`/top-k 选择），**跑不了 GQA 形状**。但在它自己的形状上与仓库内的 Triton 实现（oracle）对比：**FlyDSL 8/8 领先，前向 1.43–2.03×、反向 1.52–2.54×** —— 这就量化了立项文档"骑官方 DSL"那个决策。
6. 编译影响面：`attn/{gqa, gqa_causal, gqa_backwards, gqa_causal_backwards}` **4/4 曾失败**；`gemm/bf16fp32`、`layernorm`、`rotary`、`softmax`、`torch_scaled` **5/5 一直正常**。即库核心健康，只有 attention 这一族的写法踩到了坑。

## 0. 主结果：HK vs turbo 全后端（一张表）

**四个后端同形状、同进程、同一批张量背靠背**（每个 (shape, causal) 一个子进程；HK 的 causal / non-causal
扩展模块名都是 `tk_kernel`，无法同进程共存）。bf16（fp8 列除外），D=128，30 warmup / 30 iters，单位 **TFLOPS**。

turbo 的三条路径（`kernels/attention/attention_aiter_impl.py:10-13` 的 dispatch policy）：
`flash_attn_func(sink=None)` → **aiter csrc/CK+asm-v3**（生产默认）；`flash_attn_func(sink=<t>)` → **aiter Triton**
（**唯一**开关就是 sink）；`flash_attn_fp8_func()` → **turbo 自带 Triton blockwise fp8**（block=64）。

| shape | causal | **HK** | turbo aiter CK/asm | turbo aiter Triton | turbo Triton fp8 | 最快 turbo | **HK / 最快 turbo** | cos |
|---|:---:|---:|---:|---:|---:|:---:|:---:|---:|
| B=16 H=64/8 N=2048 | ✗ | **1057** | 983 | 744 | 411 | ck | **1.08×** | 0.999995 |
| B=8 H=64/8 N=4096 | ✗ | **1138** | 1092 | 790 | 511 | ck | **1.04×** | 0.999995 |
| B=1 H=32/8 N=4096 | ✗ | **1114** | 958 | 649 | 270 | ck | **1.16×** | 0.999994 |
| B=16 H=64/8 N=2048 | ✓ | **891** | 774 | 616 | 207 | ck | **1.15×** | 0.999996 |
| B=8 H=64/8 N=4096 | ✓ | **1051** | 953 | 721 | 314 | ck | **1.10×** | 0.999996 |
| B=1 H=32/8 N=4096 | ✓ | **820** | 764 | 447 | 148 | ck | **1.07×** | 0.999995 |

### 加速比

> **HK / 最快 turbo = 1.04 – 1.16×，几何平均 1.10×（6/6 全胜）。**

**"最快 turbo" 在全部 6 行都是 aiter CK/asm** —— Triton 与 fp8 从未赢过，所以"对最快 turbo"与"对生产默认路径"是同一个数。
turbo 三路径的稳定排序：**aiter CK/asm > aiter Triton（0.59–0.80×）> Triton fp8（0.19–0.47×）**。

### 口径限定（三条，都会影响怎么读这张表）

1. **aiter Triton 那列算的是"带 sink 的 attention"**（多一项）—— sink 是切到该 kernel 的唯一开关。所以它是"这条 kernel 路径的成本"，**不是 CK 路径的 drop-in 替代**，数值本就不同。
2. **fp8 列精度不同**。其"慢 2.4–5×"与 DSV3 报告的「fp8 attention 慢 4.0×、精度低 25 dB」一致 —— 该报告"当前只适合做正确性验证"的结论在 GQA 形状上同样成立。
3. **turbo 侧略偏保守**：`out=` shim 多一次输出拷贝（见 §1）。另 HK 写入预分配的 `out`/`lse`，turbo 内部自行分配 —— 这是两者 API 的固有差异，不是可消除的偏差。

**与论文的关系**：论文报 attention forward「1.0–2.1× AITER」。本次落在**低端**，与 08-12 GEMM 那轮结论一致 ——
**MI355X 上的 aiter 已远强于论文的基线**，论文里的大倍数不应外推到当前软件栈。

> 早先分两次运行（`bench_fwd.py` 50/50 + `bench_turbo_backends.py` 20/20）得到的两张表已被本表取代：
> 两次之间 aiter CK 列差 1–3%，跨运行拼表算比值不严谨。本表全部数字来自 `bench_all.py` 的单次背靠背测量。

## 0.2 FlyDSL vs Triton：DSV4 sparse-MLA（**另一个算子**，不可与上表并列）

**FlyDSL 在 turbo 里的 attention 只有这一个算子**：`primus_turbo/flydsl/attention/sparse_mla_{fwd,bwd}.py`
—— 单 latent MQA（K==V 取 kv 前 `kv_lora_rank` 列）+ per-token top-k KV 选择 + SWA band + 可选 per-head sink，
形状硬编码 `kv_lora_rank=512` / `d_qk=576` / `num_heads%32==0` / `topk%32==0`，且**未接进 `pytorch/ops/`**（只有 tests 调）。
**它跑不了 GQA 形状**，所以不能和 §0/§0.1 放同一张表。

但它有一个有意义的对照：仓库里 `primus_turbo/triton/attention/sparse_mla.py` 是同一算子的 Triton 实现（定位为 flydsl 的 oracle）。
所以下表既是性能对比，也是**"这个 DSL 相对 Triton 值多少"的直接测量**。输入构造与 SNR 口径复用 turbo 自己的测试。

| variant | cr | seqlen | heads | topk 宽 | fwd FlyDSL | fwd Triton | **fwd 速比** | bwd FlyDSL | bwd Triton | **bwd 速比** | fwd SNR |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---:|
| flash | 0 | 1024 | 64 | 128 | 0.065 ms | 0.110 ms | **1.70×** | 0.187 ms | 0.473 ms | **2.54×** | 74.7 dB |
| flash | 0 | 2048 | 64 | 128 | 0.116 | 0.175 | **1.51×** | 0.361 | 0.710 | **1.97×** | 76.6 |
| flash | 4 | 1024 | 64 | 384 | 0.108 | 0.193 | **1.80×** | 0.580 | 1.253 | **2.16×** | 51.5 |
| flash | 4 | 2048 | 64 | 640 | 0.268 | 0.475 | **1.77×** | 1.409 | 3.064 | **2.17×** | 51.6 |
| pro | 0 | 1024 | 128 | 128 | 0.115 | 0.176 | **1.53×** | 0.380 | 0.638 | **1.68×** | 74.5 |
| pro | 0 | 2048 | 128 | 128 | 0.216 | 0.309 | **1.43×** | 0.695 | 1.064 | **1.53×** | 76.5 |
| pro | 4 | 1024 | 128 | 384 | 0.183 | 0.339 | **1.85×** | 0.832 | 1.262 | **1.52×** | 51.5 |
| pro | 4 | 2048 | 128 | 640 | 0.453 | 0.919 | **2.03×** | 2.046 | 3.191 | **1.56×** | 51.5 |

**FlyDSL 8/8 领先：前向 1.43–2.03×、反向 1.52–2.54×**，SNR 51–77 dB（测试门槛 40 dB）。
`cr=0`（纯 SWA，闭式窗口，跳过 topk 的 HBM load）SNR 更高，符合其走的是特化路径。

**这条数据的意义**：它量化了 08-12 立项文档里「骑官方 DSL」那个决策 —— **在同一个算子上，FlyDSL 相对 Triton 值 1.4–2.5×。**
顺带一个代码质量观察：FlyDSL 的 sparse-MLA 前向在编译期抛一串 `UserWarning: Variable 'm_f' is assigned inside a control flow body … may be a closure variable captured from an outer function`（`sparse_mla_fwd.py:801`），不影响结果但说明该 DSL 的作用域检查还比较原始。

## 1. turbo 侧结果（含反向，作为基线）

`primus_turbo.pytorch.ops.flash_attn_func`，bf16，D=128，`PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=0`，30 warmup / 30 iters。
FLOPs 口径与 HK 自带 `test_python.py` 一致：`fwd = 4·B·N²·H·D / (2 if causal)`，`bwd = 2.5 × fwd`。

| workload | B | H | H_KV | N | causal | fwd ms | fwd TF | bwd ms | bwd TF |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| HK-gqa 默认（Llama3-70B 8:1） | 16 | 64 | 8 | 2048 | ✗ | 2.236 | **983** | 5.991 | **918** |
| 同上 | 16 | 64 | 8 | 2048 | ✓ | 1.409 | 781 | 3.943 | 697 |
| HK-setup MHA（16/16） | 16 | 16 | 16 | 2048 | ✗ | 0.586 | 938 | 1.512 | 909 |
| 同上 | 16 | 16 | 16 | 2048 | ✓ | 0.391 | 702 | 0.997 | 690 |
| 长序列（8:1） | 8 | 64 | 8 | 4096 | ✗ | 4.058 | **1084** | 10.503 | **1047** |
| 同上 | 8 | 64 | 8 | 4096 | ✓ | 2.300 | 956 | 6.444 | 853 |
| Llama3-8B（4:1）b=1 | 1 | 32 | 8 | 4096 | ✗ | 0.294 | 935 | 0.730 | 941 |
| 同上 | 1 | 32 | 8 | 4096 | ✓ | 0.179 | 767 | 0.476 | 722 |

**读表注意**：causal 行的 TFLOPS 低于非 causal 是口径效应——FLOPs 折半但耗时不到一半，属正常。

**与 HK 论文的交叉校准**（`papers/hipkittens.md` §4.1）：论文报 AITER 的 MHA 非 causal backward 在 seq 4096 / 8192 为 **1018 / 1169 TF**（B=16 H=16 D=128）。本次 turbo 在 GQA 8:1 / seq 4096 上测得 **1047 TF**，量级一致 → **这份 turbo 基线可当作论文中 AITER 列的本机对照**（也与 DSV3 报告"turbo ≈ TE ≈ 同一批 aiter/CK 汇编 kernel"的结论自洽）。

### turbo 侧的两处必要处理

| 处理 | 原因 |
|---|---|
| **`out=` shim** | 本 clone（`9c1e61c1`）pin 了 `aiter 0.1.14.post1`，容器自带的新 aiter 已删除 `_flash_attn_forward(out=)` → 不 shim 直接 `TypeError`（与 DSV3 报告 §4 同一现象）。改用 pin 的老 aiter 则 causal 前向背上已知的 **1.89× 缺陷**（配置 bug，非 kernel 属性），故选择 shim。shim 多一次输出拷贝 ⇒ **前向数字略偏保守** |
| **`ATOMIC_FP32=0`** | 库内默认 `"1"` 在 gfx950 上是慢反向路径（DSV3 报告 §5：反向 14.35 → 8.61 ms） |

## 2. 编译歧义的根因与修法

### 结论

**触发条件 = 「模板内 + 经别名模板书写的依赖类型」，而该别名的模板参数在展开里未被使用。**

关键别名（`kernel.cpp:91`）：

```cpp
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s>
using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;   // 注意：D 未出现在右侧
```

kernel 是 `template<int D> __global__ void attend_ker(const attn_globals<D> g)`，内部写
`attn_tile<D, float, col_l, rt_32x32_s>` 及其 `::row_vec`。**同一个类型的两种写法给出完全不同的结果**：

| 写法 | 结果 |
|---|---|
| `attn_tile<D, float, col_l, rt_32x32_s>`（依赖 `D`） | **3 个歧义** |
| `rt<float, KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_32x32_s>`（同类型，非依赖） | **0 error** |

即 clang 22 在该依赖别名上未能正确判定三个 concept 约束互斥（三者分别要求 `T::identifier` 等于
`ducks::rt::identifier` / `ducks::art::asm_identifier` / `ducks::rv::identifier`，**定义上互斥**），
导致三个重载同时被视为可行。

**这不是 ROCm 版本问题**，是 HK 的书写风格（把 tile 别名参数化到 `D` 却不使用 `D`）与本 clang 的
concept 判定实现相互作用的结果。GEMM / layernorm / softmax / rotary 不这么写，所以不受影响。

### 修法（语义等价）

pybind 只绑 `dispatch_micro<ATTN_D>`，kernel 实际只以 `ATTN_D` 实例化一次 → **把模板参数固定成常量**：

```cpp
// 原
template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {
// 改
__launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<ATTN_D> g) {
    constexpr int D = ATTN_D;
```

外加把 `attend_ker<D>` 的两处引用改为 `attend_ker`。脚本：`/perf_apps/xiaoming/scratch/hk_attn/patch_nodep.py`
（正则匹配模板头，不动 HK 仓库，产物写到 `p_{noncausal,causal}/`）。

编译结果：**0 error**，VGPRs 238 / AGPRs 0 / **Occupancy 2 waves/SIMD**（= 8 waves 单 WG 占满 CU）/ 无 spill / LDS 动态。

### 症状（修复前）

```
kernel.cpp:161:5: error: call to 'zero' is ambiguous
  candidate: include/cdna4/ops/warp/register/tile/maps.cuh:375
             template<ducks::rt::all T>  void zero(T &dst)
  candidate: include/cdna4/ops/warp/register/tile/assembly/maps.cuh:335
             template<ducks::art::all T0> void zero(T0 &dst)
  candidate: include/cdna4/ops/warp/register/vec/maps.cuh:97
             template<...> void zero(T &dst)
  [with T = kittens::rv<float, 32, 32, rt_shape<32,32,4>, ducks::rv_layout::ortho>]
```

同类错误还有 `ones` / `copy` / `mul` / `sub` / `exp2`，单文件 **20 个 error**。

### 影响面：全仓 `kernels/cdna4/` 逐目标扫描（2026-08-14 13:35）

| 目标 | error 数 | 首条错误 |
|---|---:|---|
| `attn/gqa`（前向非 causal） | **20** | `kernel.cpp:161:5: call to 'zero' is ambiguous` |
| `attn/gqa_causal`（前向 causal） | **20** | `kernel.cpp:247:5: 同` |
| `attn/gqa_backwards`（反向非 causal） | **20** | `attn_fwd_non_causal.cpp:113:5: 同` |
| `attn/gqa_causal_backwards`（反向 causal） | **20** | `attn_fwd_causal.cpp:201:5: 同` |
| `gemm/bf16fp32` | **0** ✓ | — |
| `layernorm` | **0** ✓ | — |
| `rotary` | **0** ✓ | — |
| `softmax` | **0** ✓ | — |
| `torch_scaled` | **0** ✓ | — |

另 `training/llama/csrc/attn_fwd_causal.cpp`（训练 harness 版）同样 ✗。

→ **精确结论：attention 全族（4/4）断，其余全族（5/5）通。** 不是"HK 编不过"，是**只有 attention 编不过**。
库核心（tile 原语、LDS、MFMA 封装）是健康的 —— `layernorm` / `softmax` / `rotary` 也用 `rv` 向量却能通过，说明触发条件比"用了 `rv`"或"用了 32×32 形状"更窄（最小复现亦通过，见下表）。

### 已排除的原因

| 假设 | 检验 | 结论 |
|---|---|---|
| attn Makefile 的额外 flags（`-ffast-math` / `--save-temps` / `-DHIP_ENABLE_WARP_SYNC_BUILTINS`） | 用 GEMM 的最小 flags 重编 | ✗ 仍 20 errors |
| include 重组（`f2b97dd6` "Unify kernel build plumbing…"）引入 | 克隆并回退到其父 `cd090ae9`（2026-05-17，布局为 `kernels/attn/`） | ✗ 同类错误；且 `tile/assembly/` 在重组前**已存在** |
| 上游已修 | `git fetch origin main` | ✗ `origin/main == HEAD == a288366e`，**无更新可用** |
| attn 比 GEMM 多包了头文件 | 对比两者开头 | ✗ 都只有 `kittens.cuh` + `pyutils/pyutils.cuh` |
| 概念定义重叠（`ducks::rt::all` vs `ducks::art::all`） | 读 `rt.cuh:106` / `art.cuh:250` / 各类型的 `identifier` | 三个 identifier 互不相同、**定义上互斥** —— 所以问题在"判定"而非"定义" |
| 「32×32 形状本身」触发 | 最小复现 `repro_ambiguous.cpp`：`rt_fl<64,32,col,32x32x4>` + `rv<float,32,32,32x32x4,ortho>` 上调 `zero`/`ones` | ✗ **0 error** |
| pyutils / 自定义 `exp2` / `rv_all_below` | 逐个移除（注意：先前一次"把 `exp2` 移入 namespace"是**无效测试**，因为随后 `using hkfix::exp2;` 又把它拉回全局作用域） | ✗ 均仍 3 歧义 |
| 单纯"模板 + 依赖类型" | `repro2.cpp` HK_CASE=2（自建别名 `attn_tile<D> = rt<float,64,D/4,…>`，**D 被使用**） | ✗ 0 error → **关键差异是别名里 `D` 未被使用** |

### 定位过程（有效的那次二分）

保留 kernel 前言 + 把 3 个调用放到不同上下文：

| 变体 | 结果 |
|---|---|
| 前言到第 91/100 行 + **全局非模板** probe | **0 error** |
| 前言到第 104 行（= 放进 `attend_ker<D>` 内部）+ 同样 3 个调用 | **3 歧义** |
| 同上，但类型改为非依赖的 `rt<float,64,32,col_l,rt_32x32_s>` | **0 error** |

三步把范围从"整个文件"收敛到"依赖别名写法"这一点。

## 3. 解读

1. **这是一个关于 HK 维护状态的实质信号。** attention 是 HK 论文的头号结果（GQA backward 超基线 1.8–2.5×），而它在当前 ROCm 上**开箱编不过**，且上游 main 无修复。此前 08-12 的 GEMM 实测已发现 HK 的另一类脆弱性（`WARPS_M/WARPS_N` 改了会静默算错、`BLOCK_SIZE=64` 过不了静态断言）。**两次独立观察都指向同一件事：HK 的参数化与可移植性弱于其论文给人的印象。**
2. **对 peer-tiles 立项的影响：不改变结论，但加重一条风险。** 立项文档已把 HK 定位为「纪律与技法的来源」而非代码库（[§0.4](./2026-08-12_1530_repo-charter-and-primitive-api.md)）。本轮进一步说明：**连"取用它的代码"这一步都有工具链成本**，所以"技法迁移"的定位比"代码复用"更稳。
3. **turbo 基线仍然有用。** 它与论文的 AITER 列吻合，所以论文里 HK-vs-AITER 的相对倍数可以**估**出 HK 在本机的水平（但那是估计，不是测量，不可用于任何结论）。

## 4. 下一步

| 优先级 | 动作 | 说明 |
|---|---|---|
| **P0** | **反向对比**：同样 patch `gqa_backwards` / `gqa_causal_backwards`，与 turbo 反向对齐形状 | **论文最强的主张在反向**（GQA 超基线 1.8–2.5×）；turbo 反向基线已在 §1；反向有 3 个模块（`tk_kernel_bkwd` / `_prep` / `_fwd`）与不同签名，工作量比前向大 |
| P1 | 给上游开 issue/PR | 本篇已有完整最小复现与语义等价修法；对 HK 是真实改进（把 tile 别名的未使用参数去掉，或在 kernel 内改非依赖写法） |
| P2 | 扫更多 GQA 比例（16:1 = Qwen3-235B）与 head_dim=64 | HK 有 `kernel_d64.cpp`；16:1 需 `ATTN_H=64 ATTN_H_KV=4` 重编 |
| P2 | 若要引用绝对值，加 keepalive 钉时钟 + 多轮取 min | 本项目 08-12 已证本机时钟单向下漂；**比值（背靠背）可信，绝对值不可跨会话比** |

## 复现

```bash
ssh smci355-ccs-aus-n04-21
# turbo 基线（可用）
docker exec xiaoming-dev bash -lc '
  cd /perf_apps/xiaoming/scratch/hk_attn && HIP_VISIBLE_DEVICES=0 python3 bench_turbo.py'

# HK 编译失败复现
docker exec xiaoming-dev bash -lc '
  export ROCM_PATH=/opt/rocm
  cd /perf_apps/xiaoming/slab/3rd/hk/kernels/cdna4/attn/gqa
  make ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048 ATTN_D=128 2>&1 | grep -c "error:"'

# 最小复现（干净通过，用于说明不是形状问题）
docker exec xiaoming-dev bash -lc '
  cd /perf_apps/xiaoming/scratch/hk_attn
  /opt/rocm/bin/hipcc repro_ambiguous.cpp -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 \
    -I/perf_apps/xiaoming/slab/3rd/hk/include -I/opt/rocm/include/hip -c -o /dev/null'
```

脚本：`/perf_apps/xiaoming/scratch/hk_attn/{bench_turbo.py, bench_fwd.py, repro_ambiguous.cpp}`
（`bench_fwd.py` 是 HK-vs-turbo 的对比 harness，已写好但因 HK 编不过未能运行）
回退版本的克隆：`/perf_apps/xiaoming/scratch/hk_attn/hk_prereorg`（`cd090ae9`，可删）

## 相关

- [HK GEMM 实测（08-12）](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md) —— 同类脆弱性的另一组证据
- [立项文档 §0.4](./2026-08-12_1530_repo-charter-and-primitive-api.md) —— HK 的角色定位
- [`papers/hipkittens.md`](../../papers/hipkittens.md) §4.1（register pinning 的 855→1024 TF）· 文首环境（ROCm 7.0 preview）
- `/perf_apps/xiaoming/MegaMoE/docs/attention_dsv3_8k_backends.md` —— aiter pin 与 `ATOMIC_FP32` 两个配置问题的原始报告
