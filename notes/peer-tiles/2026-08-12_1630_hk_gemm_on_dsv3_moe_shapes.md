# HK bf16 GEMM 在 DSV3 MoE 形状上的实测：出厂配置弱，调完反超

> **When**: 2026-08-12 16:30 UTC+8（**17:45 增补 §调优，主结论反转**）
> **Where**: `smci355-ccs-aus-n04-21`（8× MI355X / gfx950），容器 `xiaoming-dev`（`rocm/primus:v26.3`，ROCm 7.2.1 / hipcc clang-22 / torch 2.10.0+git94c6e04），单卡 `HIP_VISIBLE_DEVICES=0`
> **Context**: 验证「HK 有高性能 GEMM，所以更适合 mega kernel」这个前提。**出厂配置下不成立，逐形状调优后成立**，见 [立项文档 §0.4](./2026-08-12_1530_repo-charter-and-primitive-api.md)

## TL;DR

**⚠ 本篇有两层结论，第二层推翻了第一层的判断（不是数据）：**

**第一层（出厂配置，`BLOCK_SIZE=256`）**：

1. per-expert 真实形状（M=2048）上 HK 打平偏输：FC1 gate_up **0.86×**、FC2 down 1.07×、FC1 dgrad 1.02×、FC2 dgrad **0.70×**（分母为 torch/hipBLASLt）。
2. 分 chunk 后崩塌：M=1024 **0.68×** → M=512 **0.49×** → M=256 **0.37×**。
3. 机制是**网格饥饿**，不是内循环慢。tile 硬编码 `BLOCK_SIZE=256`、`grid()=(N/256)*(M/256)`；实测吞吐 ≈ (grid WG 数 / 256 CU) × 峰值。M=256 时只有 16 个 WG 撒在 256 个 CU 上。

**第二层（逐形状调优后，见 [§调优](#调优2026-08-12-1745逐形状取最优后结论反转)）**：

4. **只改两个常量（tile 256→128、chiplet window WGM）就把弱格全部拉起来**：chunk2 **0.68→1.31×**、FC2 dgrad **0.70→1.28×**、chunk4 **0.49→1.07×**、chunk8 **0.37→0.84×**。**5 个 MoE 形状里 HK 赢 3 个。**
5. **但没有通吃配置**：BS=128 让大方阵从 ~1.0× 掉到 0.69–0.75×。**这对 mega kernel 恰好不是问题——形状是编译期已知的，本来就该逐点特化**，而这正是 HK「一份手调配置」模型的适配场景，也是 hipBLASLt 运行时选型优势失效的地方。
6. 有效旋钮只有 **3 个**：`BLOCK_SIZE`、`WGM`（chiplet window，chunk2 上 +21%，对应论文的 +19%）、以及大形状留 BS=256。`WARPS_M/WARPS_N` **不是旋钮**（一改数值就错，store 索引写死 2×4）；`BLOCK_SIZE=64` 编译失败（静态断言）。

**两层都成立的部分**：

7. **HK 仍然没有 grouped / 变长-M GEMM，只有 dense**。mega kernel 里 32 个 expert 共用常驻 grid，要用 HK 的 tile 原语就得自己写 grouped driver——而 gen-3 的 FlyDSL `gemm_helper.py` 已经有了（grouped mxfp8 wgrad 2007–2199 TFLOPS）。
8. 噪声：run-to-run 大形状 ~2%、小形状最多 6.8%，**且单向下漂**（idle sclk 掉到 232 MHz）。8192³ 的 1.02× 落在漂移内 —— HK 在大方阵上是**追平** hipBLASLt（与论文自述一致）。

## 背景

`notes/peer-tiles/` 的立项前提之一是「HK 提供库级 GEMM tile，可以架在 mega kernel 里」。此前一轮已发现载体判断错了（gen-3 是 100% FlyDSL，不是 HIP C++），但 HK 的 GEMM 在 MoE 形状上到底行不行仍是推测。本篇实测。

## 做了什么

**编译**（4.9 s）：

```bash
cd /perf_apps/xiaoming/slab/3rd/hk/kernels/cdna4/gemm/bf16fp32   # HK @ a288366e
export ROCM_PATH=/opt/rocm && make clean && make                 # SRC=256_256_64_32_with16x32.cpp
```

编译器资源报告（`-Rpass-analysis=kernel-resource-usage`）：

| 项 | 值 |
|---|---|
| VGPRs | **210** |
| AGPRs | **0** |
| SGPRs | 53 |
| Occupancy | **2 waves/SIMD** |
| Spill (SGPR / VGPR) | 0 / 0 |
| ScratchSize | 0 |

即这是**编译器托管**的版本，没有用 register pinning（AGPR=0）。

**HK 自带 `bench.py` 跑不了**：第 3 行 `from aiter.tuned_gemm import tgemm` → 容器内 aiter 依赖 flydsl 且版本校验失败：

```
ImportError: Unsupported `flydsl` version: expected `0.1.1.dev409`, got `0.2.4`.
```

与 HK 本体无关（HK 只要 torch + hipcc）。改用自写驱动跳过 AITER 腿：

- `/perf_apps/xiaoming/scratch/hk_gemm_bench.py` —— HK 自带 8 个方阵形状，复用 HK 的 `utils.bench_gemm`（500 warmup / 500 iter）
- `/perf_apps/xiaoming/scratch/hk_gemm_dsv3.py` —— DSV3 MoE 形状，200 warmup / 200 iter，**每形状内 HK 与 torch 背靠背**

**方法学**：布局沿用 HK 自带 bench 的约定 —— HK 走 NT（B 为 `[n,k]`），torch 走 NN（B 为 `[k,n]`）。计时用 `torch.cuda.Event` + 每次 `synchronize`。因为观察到会话内时钟单向下漂，DSV3 那轮改成**同形状内两个后端背靠背**，以抵掉漂移。

**形状推导**（DSV3：H=7168, I=2048, E=256, topk=8, EP=8 → 32 local experts，T=8192/rank）：
收到的 token-expert 对 = 8192×8 = 65536 → 每 expert `M = 65536/32 = 2048`；若按 src GPU 切 8 chunk → 每 (src,expert) `M = 256`。

> **`chunk8` / `chunk4` / `chunk2` 不是独立算子** —— 它们**就是 FC1 gate_up**（`n=2I=4096`、`k=H=7168` 与 `FC1 gateup` 行相同），只是 M 从 2048 切成 256 / 512 / 1024。含义是"若 mega kernel 为把通信藏进计算而一个 src 到了就开算，每次 GEMM 调用看到的 M"。
>
> 两条限定：(a) 这模拟的是 **gen-1 的按 src/chunk 切分结构**，不是 gen-3 的「一次通信 × 一个 GEMM」stage 融合（后者 GEMM 保持满尺寸 tile 几何）；(b) **单独 launch 的测法会放大网格饥饿** —— 真实 mega 里 32 expert × 8 chunk 在同一常驻 grid 内并发，总 tile 数远多于此。本测法能干净隔离 tile 几何的影响，但高估并行度问题，故 grouped GEMM 对照列为 P0。

## 结果

### 1. HK 自带方阵形状（两轮，看漂移）

| m,n,k | HK run1 | HK run2 | torch run1 | torch run2 | HK/torch (run1) |
|---|---:|---:|---:|---:|---:|
| 8192,8192,8192 | 1516 | 1507 | 1484 | 1478 | 1.02× |
| 4096,8192,2048 | 1277 | 1254 | 1184 | 1172 | 1.08× |
| 8192,4096,2048 | 1273 | 1251 | 1174 | 1162 | 1.08× |
| 8192,2048,4096 | 1359 | 1335 | 1260 | 1248 | 1.08× |
| 4096,4096,4096 | 1359 | 1333 | 1291 | 1276 | 1.05× |
| 2048,4096,4096 | 941 | 877 | 1030 | 1012 | **0.91×** |
| 2048,2048,4096 | 492 | 475 | 716 | 696 | **0.69×** |
| 2048,2048,2048 | 421 | 403 | 537 | 512 | **0.78×** |

run2 相对 run1 **全部为负**（HK −0.6% ~ −6.8%，torch −0.4% ~ −4.7%）→ 单向下漂而非随机噪声。参考：论文报 8192³ = 1610 TFLOPS（ROCm 7.0 preview），本机 1516。

### 2. DSV3 MoE 形状（HK 与 torch 背靠背，单位 TFLOPS）

| workload | m | n | k | HK | torch | HK/torch |
|---|---:|---:|---:|---:|---:|---:|
| fwd FC1 gate_up `[M,H]@[H,2I]` | 2048 | 4096 | 7168 | 996 | 1152 | **0.86×** |
| fwd FC2 down `[M,I]@[I,H]` | 2048 | 7168 | 2048 | 1100 | 1030 | 1.07× |
| bwd FC1 dgrad `[M,2I]@[2I,H]` | 2048 | 7168 | 4096 | 1268 | 1249 | 1.02× |
| bwd FC2 dgrad `[M,H]@[H,I]` | 2048 | 2048 | 7168 | 516 | 740 | **0.70×** |
| dW1 形状 ※ | 4096 | 7168 | 2176 | 1154 | 1138 | 1.01× |
| dW2 形状 ※ | 7168 | 2048 | 2176 | 1132 | 1011 | 1.12× |
| **chunk8 FC1** M=256 | 256 | 4096 | 7168 | **130** | 354 | **0.37×** |
| **chunk4 FC1** M=512 | 512 | 4096 | 7168 | **260** | 533 | **0.49×** |
| **chunk2 FC1** M=1024 | 1024 | 4096 | 7168 | **508** | 748 | **0.68×** |

※ wgrad 两行只是"同形状的普通 GEMM"。真实 wgrad 是 `A^T@B`，`dispatch_micro` 接不了，**不是 drop-in 对照**，仅供看该长宽比下的吞吐。

## 解读

### 网格饥饿是唯一需要的解释

`256_256_64_32_with16x32.cpp:5` `constexpr int BLOCK_SIZE = 256`，`:33` `grid() = (N/BLOCK_SIZE)*(M/BLOCK_SIZE)`。固定 256×256 输出 tile ⇒ WG 数 = (M/256)×(N/256)。对 N=4096 固定、扫 M：

| M | grid WG | 占 256 CU | 1516 × 占比 | 实测 |
|---:|---:|---:|---:|---:|
| 256 | 16 | 6.3% | 95 | **130** |
| 512 | 32 | 12.5% | 190 | **260** |
| 1024 | 64 | 25% | 379 | **508** |
| 2048 | 128 | 50% | 758 | **996** |

**实测几乎就是"grid 占 CU 比例 × 峰值"**，且 HK 的绝对吞吐随 M 近似线性（130→260→508→996 对应 M 翻倍）。hipBLASLt 在同形状换小 tile，M=256 拿到 354（2.7× 于 HK）。

所以 HK 在 MoE 形状上的弱势**不是内循环质量问题**——它的内循环在大方阵上追平 hipBLASLt。是**单一硬编码 tile + 无自动选型**。这与论文自己的批判点同类（`papers/hipkittens.md` §9 批判 5：chiplet swizzle 的 W/C 需按形状调参，论文没给自动调参器）。

### 对 HK 公平的保留：这个测法高估了问题

上表是**每个 expert 单独 launch**。真实 mega kernel 里 32 个 expert 在**同一个常驻 grid** 内完成，总 tile 数恢复到 32×16 = 512，足以喂满 256 CU ⇒ 网格饥饿不一定平移过去。

**真正该测的是 grouped GEMM（每 expert 变长 M、一次 launch）——而 HK 没有 grouped GEMM，只有 dense。** 这才是缺口的准确位置：

> 要把 HK 的 GEMM 用进 mega kernel，缺的不是更快的内循环，是**一个 grouped / 变长-M 的 driver 架在它的 tile 原语之上**。

而这件事 gen-3 已经做完了：`primus_turbo/flydsl/utils/gemm_helper.py`（1496 行）+ `fp8/gemm_mxfp8_tile.py`，grouped mxfp8 wgrad 实测 **2007–2199 TFLOPS**（= 自测上限的 85%）。

### 结论：把立项文档 §0.4 再夯一层

本轮实测支持上一轮的修正 —— **HK 能贡献的是纪律与技法（register pinning / 角色轮换 / LDS phase 表 / 多 MFMA 形状共存），不是 GEMM 实现本身。** 而且新增一条证据：编译出来的 kernel 是 **AGPR=0 / VGPR=210 / 2 waves/SIMD** 的编译器托管版本，连 HK 自己的 register pinning 在这个 GEMM 里都没启用。

---

## 调优（2026-08-12 17:45）：逐形状取最优后结论反转

### 做了什么

把 HK kernel 的 5 个 `constexpr` + 函数内的 `WGM` 生成一份 `-D` 可覆盖的副本（不回写 HK 仓库），扫配置。**每个变体先验数值再计时**（相对 Frobenius 误差 < 2e-2 对 `A@B.T` 的 fp32 参考），HK 与 torch 每形状内背靠背。

- `/perf_apps/xiaoming/scratch/hk_tune/gen_variant.py` —— 生成参数化副本（`HK_BS` / `HK_KSTEP` / `HK_WM` / `HK_WN` / `HK_DOT` / `HK_WGM`）
- `/perf_apps/xiaoming/scratch/hk_tune/bench_weak.py` —— 数值校验 + 弱形状计时（`HK_SKIP_LARGE=1` 跳过大形状）

### 哪些是真旋钮

| 旋钮 | 结果 |
|---|---|
| `BLOCK_SIZE` 256 → **128** | **有效，且是主要杠杆**。见下表 |
| `WGM`（chiplet window，论文 Algorithm 1 的 `W`） | **有效**。chunk2 上 WGM 8→2 是 826→**996 TFLOPS（+21%）**，对应论文报的 +19% |
| `BLOCK_SIZE` = 64 | **编译失败**：`rows % rt_base::rows == 0` 静态断言 + `zero-length arrays are not permitted in HIP device code` |
| `WARPS_M` / `WARPS_N` | **不是旋钮**。改成 2×2 或 4×2 都能编过但**数值全错**（rel ≈ 1.0）——store 的索引数学写死了 2×4 |

### 逐形状最优（数值全部 ok）

| 形状 | 最优配置 | HK | torch | 比值 | 出厂配置 |
|---|---|---:|---:|---:|---:|
| chunk8 `256×4096×7168` | BS128 WGM4 | 307 | 366 | **0.84×** | 0.37× |
| chunk4 `512×4096×7168` | BS128 WGM2 | 588 | 547 | **1.07×** | 0.49× |
| chunk2 `1024×4096×7168` | BS128 WGM2 | **996** | 760 | **1.31×** | 0.68× |
| FC2 dgrad `2048×2048×7168` | BS128 WGM16 | 964 | 753 | **1.28×** | 0.70× |
| FC1 gateup `2048×4096×7168` | BS128 WGM2 | 1038 | 1174 | **0.88×** | 0.86× |
| LARGE `4096×4096×4096` | **BS256** WGM8 | 1357 | 1317 | 1.03× | — |
| LARGE `8192×8192×8192` | **BS256** WGM8 | 1532 | 1543 | 0.99× | — |

### 此消彼长，且这对 mega 不是问题

`BLOCK_SIZE=128` 在大形状上**回退**：4096³ 1.03× → **0.75×**，8192³ 0.99× → **0.69×**。这与论文的机制说法一致（大 tile 提算术强度，小 tile 提并行度；小 M 被迫走小 tile 一侧）。

**没有通吃配置 → hipBLASLt 靠运行时按形状选 tile 取胜，HK 只出厂一份配置。** 但在 mega kernel 里，**形状是编译期已知的**：FC1 / FC2 / dgrad / 各 chunk 尺寸各自实例化自己的 tile 即可。**hipBLASLt 的运行时选型优势在这里失效，而 HK「一份手调配置 + 编译期特化」的模型恰好对口。**

### 剩下两个输的格子，原因不同

| 格子 | 比值 | 原因 | 能不能修 |
|---|---|---|---|
| chunk8 `M=256` | 0.84× | **仍是网格饥饿**：BS=128 时 grid = (4096/128)×(256/128) = **64 WG = 256 CU 的 25%** | 要**非对称 tile**（如 BM=64 / BN=128 → 128 WG）。当前副本是单一 `BLOCK_SIZE` 且 BS=64 过不了断言；但 `micros/192x256/kernel.cpp` 已是 `BLOCK_SIZE_M`/`BLOCK_SIZE_N` 分离 —— **机制在仓库里存在，换基底文件即可试** |
| FC1 gateup `2048×4096×7168` | 0.88× | **不是网格饥饿**：BS=128 时 grid = 32×16 = **512 WG，2 WG/CU 满占用**。这是真实的内循环效率差（K=7168 长、N=4096），torch 1174 vs HK 最好 1038 | 未知。怀疑 torch 走 split-K 或更好的 K 流水。**这是唯一一个 HK 因内循环而非并行度落后的格子，值得单独 rocprof** |

### 对结论的影响

**「HK 的 GEMM 不适合 mega」这个判断作废**，替换为：

> HK 的 GEMM 在 MoE 形状上**出厂配置弱（0.37–0.86×），逐形状调 tile + chiplet window 后反超（3/5 格 1.07–1.31×）**。它缺的不是性能，是 (a) 自动选型（论文自己批判点 5 已承认）与 (b) grouped / 变长-M driver。前者在 mega kernel 里不需要（编译期特化），后者需要自己写。

---

## 下一步

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | 用 `micros/192x256/kernel.cpp`（已有 `BLOCK_SIZE_M`/`BLOCK_SIZE_N` 分离）做**非对称 tile** 扫描，目标 chunk8 `M=256` | 唯一还在网格饥饿里的格子；BM=64/BN=128 应给 128 WG（现 64） |
| **P0** | **grouped GEMM 对照**：每 expert 变长 M、一次 launch，vs gen-3 的 FlyDSL grouped mxfp8 | HK 没有 grouped，这是"能不能进 mega"的真正判据；per-expert dense 只是代理指标 |
| P1 | FC1 gateup `2048×4096×7168` 单独 rocprof | **唯一非网格饥饿的落后格**（满占用下 0.88×），怀疑 torch 走 split-K；这决定 HK 的内循环有没有真短板 |
| P2 | 若要引用绝对值，加 keepalive 钉时钟 + 多轮取 min | 本轮已证单向下漂；比值（背靠背）可信，绝对值不可跨会话比 |
| — | ~~**不做**：把 HK 的 dense GEMM 塞进 mega kernel~~ | **17:45 作废**——调优后 3/5 格反超，此路重新打开 |

## 复现

```bash
ssh smci355-ccs-aus-n04-21
docker exec xiaoming-dev bash -lc '
  cd /perf_apps/xiaoming/slab/3rd/hk/kernels/cdna4/gemm/bf16fp32
  export ROCM_PATH=/opt/rocm && make clean && make
  HIP_VISIBLE_DEVICES=0 python3 /perf_apps/xiaoming/scratch/hk_gemm_bench.py   # HK 自带方阵
  HIP_VISIBLE_DEVICES=0 python3 /perf_apps/xiaoming/scratch/hk_gemm_dsv3.py    # DSV3 MoE 形状
'
```

脚本：`/perf_apps/xiaoming/scratch/hk_gemm_{bench,dsv3}.py`（跳过 aiter 腿，原因见「做了什么」）。
