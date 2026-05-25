# 2026-05-21 13:30  M0 BASELINE — `mfma_tile.h` 移植 + 单 GPU GEMM bench

> 时间: 2026-05-21 13:30 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, mi355-gpu-7), 1 节点 / SLURM job 13489
> 容器: xiaoming-dev (podman, ROCm 7.2.0)
> 软件: hipcc / amdclang 22.0.0 / cmake 4.x / PyTorch 2.10
> 代码: 新 commit (未 push), `csrc/include/rocmoe/{types.h,lds_layout.h,mfma_tile.h,gemm.h}` + `csrc/gemm.hip` + `tests/test_gemm.hip` + `benchmarks/bench_gemm.hip` + `scripts/dev_on_node.sh` + `baselines/pt_rccl_moe.py` + `CMakeLists.txt`
> 原始日志: `build/Testing/Temporary/LastTest.log` (mi355-gpu-7)

## TL;DR

M0 BASELINE 落地：把 MonolithEP `mfma_tile.h` (1113 行, 99.3% MFMA reference) 直接 cherry-pick 到 `csrc/include/rocmoe/`，去掉 Layout-E 依赖（MPE/MRS/MAX_M_BLOCKS_PER_EXPERT），加最小 wrapper kernel，5 个 correctness PASS + 单 GPU bench **8192×4096×7168 = 1290 TFLOPS (99% of MI355X BF16 peak 1300 TFLOPS)**, 4096×4096×7168 = 1216 TFLOPS。skill `rocmoe-dev-loop` + `scripts/dev_on_node.sh` workflow 跑通，build/test/bench 三连绿。可以进入 M1 (Layout-P + receiver-pull dispatch)。

## 1. 上下文 / 目标

设计文档 [`./2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md) §5 路线图: **M0 = bootstrap repo + cherry-pick `mfma_tile.h` from MonolithEP, standalone GEMM bench, 验收 DSV3 grouped GEMM ≥ 950 TFLOPS / GPU**。本 note 记录 M0 完成情况。

同时落实 4 条 dev-loop 规则到 skill `rocmoe-dev-loop` (`~/workspace/slab/.cursor/skills/rocmoe-dev-loop/SKILL.md`)：

1. PyTorch+RCCL baseline 同 workload 同步报数字
2. 每 phase 都有 ctest 用例
3. 每轮优化写带 `UP/DOWN/FLAT/CRASH/WRONG/BASELINE` flag 的 note
4. build/run 全在 squeue 节点的 xiaoming-dev container 内

## 2. 做了什么

### 2.1 仓库骨架 (镜像 RocMoE-bak)

```
csrc/include/rocmoe/   # 公共 header
  types.h              # bf16_t + ROCMOE_HIP_CHECK macro (12 行)
  lds_layout.h         # MFMA 常量 + GemmLds 结构 (107 行, 自含, 不依赖 Layout-E)
  mfma_tile.h          # 直接拷 MonolithEP, 1113 行, 仅改 namespace v1->rocmoe + include 路径 + macro 前缀 MONO_->ROCMOE_
  gemm.h               # M0 standalone GEMM API (35 行)
csrc/gemm.hip          # M0 wrapper kernel (一 tile/WG, 70 行)
tests/test_gemm.hip    # bf16 vs naive bf16 matmul, max-rel ≤ 5%
benchmarks/bench_gemm.hip  # 单 GPU bf16 GEMM 微基准, hipEvent 计时
baselines/pt_rccl_moe.py   # PyTorch+RCCL MoE baseline 骨架 (M3 接 e2e bench 时启用)
scripts/dev_on_node.sh     # build/test/bench/shell 都走 ssh + podman exec
CMakeLists.txt             # gfx950 默认, Release/-O3/-ffast-math
```

### 2.2 mfma_tile.h port — 4 处编辑

| # | 内容 |
|---|---|
| 1 | header 注释改成 `RocMoE-v2 MFMA GEMM tile primitive`, 注明 cherry-pick 自 MonolithEP |
| 2 | `#include "monolith/lds_layout.h"` + `#include "monolith/v1_common.h"` -> `#include "rocmoe/lds_layout.h"` (并加 `#include <cstdint>` `<cstring>`，否则 device 端 `std::memcpy` 找不到) |
| 3 | `namespace v1 { ... }` -> `namespace rocmoe { ... }` |
| 4 | `MONO_MFMA_K_LOOKAHEAD` / `MONO_LDS_BUFS` macro 前缀 -> `ROCMOE_*` |

文件 body 一行未改。

### 2.3 lds_layout.h — 自含化

去掉 MonolithEP 的 Layout-E 依赖：
- 删掉 `SortLds` (per-expert 计数排序结构, RocMoE-v2 改 Layout-P 不需要)
- 删掉 `MAX_M_BLOCKS_PER_EXPERT` 静态断言（这个常量出在 `v1_common.h` 的 MPE，跟 Layout-E 绑死）
- 保留 GEMM tile 相关一切：`M_TILE/N_TILE/K_TILE`, `MFMA_K`, `K_STEPS`, `LDS_PAD=0`, `A/B_LDS_STRIDE`, wave 映射 (`WAVE_M=64`, `WAVE_N=64`, `MFMA_PER_WAVE_M=2`)，`GemmLds` 结构 (双 buffer, 64 KB, 2 WG/CU)
- macro `MONO_*` 全部改 `ROCMOE_*`

### 2.4 gemm.hip — 最小 wrapper 调用 mfma_tile

- 一 WG / 一 (m_tile, n_tile)，grid = m_tiles × n_tiles
- `extern __shared__ char shared_mem[]` 动态 LDS, `kernel_lds_bytes()` = 64 KB
- 关键 bug 修了 1 个 (见 §3)
- `__launch_bounds__(WG_SIZE=256, 2)` 让 LDS budget 64 KB × 2 = 128 KB 落到 1 个 CU 的 160 KB 内，2 WG/CU 占用

### 2.5 build / test / bench 跑通流程

skill 规则 4 (build inside container on a SLURM node) 通过 `scripts/dev_on_node.sh` 实现：

```bash
bash scripts/dev_on_node.sh build   # cmake -S . -B build && cmake --build build -j
bash scripts/dev_on_node.sh test    # ctest --output-on-failure
bash scripts/dev_on_node.sh bench M0 [M N K]  # 单 GPU GEMM bench
bash scripts/dev_on_node.sh shell   # 交互 shell
```

脚本本身在 login 机上跑，内部 ssh 到 squeue 取的节点，podman exec 进 `xiaoming-dev`。环境变量 `ROCMOE_NODE` 可强制覆盖节点。

## 3. 遇到的坑 / 修法

### 3.1 `std::memcpy` 在 device 代码里找不到

`mfma_tile.h` 的 `store_acc_block*` 用 `std::memcpy(&bits, &v, sizeof(bits))` 把 `__hip_bfloat16` 的 16 bit 取出来。MonolithEP 自己的版本通过 `v1_common.h` 间接 `#include <cstring>` 引入。我 trim 掉 `v1_common.h` 后没带上，编译报：

```
error: no member named 'memcpy' in namespace 'std'
```

**修法**: 在 `mfma_tile.h` 顶部直接 `#include <cstring>` (顺手加 `<cstdint>` 也补全了)。

### 3.2 单 tile PASS, 多 (m, n) tile FAIL —— c_out 没加 n_start 偏移

`store_acc_block` 写的是 `c_out[row * stride + n_off_within_tile + col]`，其中 `n_off_within_tile = wn + n*32`，永远在 [0, N_TILE) 范围。MonolithEP 的 caller 在 `gemm.hip:371` 把 `c_out` 一次性算成 `&fc1_buf[(e * MPE + m_start) * (2*F) + n_start]`（含 column 偏移），然后 mfma_tile 内 store 用 within-tile 坐标。

我第一版 wrapper 只 advance 了 row：
```cpp
uint16_t* C_u = reinterpret_cast<uint16_t*>(C + (size_t)m_start * c_stride);  // BAD
```

结果多 (m, n) tile 都把数据写到 `C[..][0:N_TILE)`，互相覆盖。`max_rel=1363, n_bad=87%`，跟随机一样。

**修法**:
```cpp
uint16_t* C_u = reinterpret_cast<uint16_t*>(C + (size_t)m_start * c_stride + n_start);
```

加这一项 fix 之后 5 个 ctest 全 PASS。**这个细节是 mfma_tile.h 调用约定上最容易踩的陷阱**，值得在 `gemm.h` 注释里反复强调（已加）。

调试方法是开了 5 个梯度 case：

| case | M | N | K | 排查的维度 |
|---|---|---|---|---|
| single_tile  | 128 | 128 | 64  | 单 tile, 单 K-tile (sanity) |
| kmulti       | 128 | 128 | 128 | 单 (m,n)-tile, 多 K-tile |
| kmulti_long  | 128 | 128 | 256 | 同上, K=256 |
| mn_multi     | 256 | 256 | 64  | 多 (m,n)-tile, 单 K-tile |
| full         | 512 | 512 | 512 | 多 (m,n)-tile, 多 K-tile |

`mn_multi` FAIL 直接定位到 column 偏移问题。

### 3.3 容器内 `$HOME` 不是 `/shared/...`

第一版 `dev_on_node.sh` 用 `cd $HOME/workspace/RocMoE`，container 里 `$HOME=/root` 没有 `workspace` 子目录。**修法**: 用绝对 NFS 挂载路径 `cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE`（NFS 在 host 和 container 内挂载点一致）。

## 4. 结果

### 4.1 ctest 全绿

```
1/5 Test #1: test_gemm_single_tile (128x128x64) ......   Passed   0.26s
2/5 Test #2: test_gemm_kmulti      (128x128x128) .....   Passed   0.26s
3/5 Test #3: test_gemm_kmulti_long (128x128x256) .....   Passed   0.27s
4/5 Test #4: test_gemm_mn_multi    (256x256x64) ......   Passed   0.26s
5/5 Test #5: test_gemm_full        (512x512x512) .....   Passed   0.27s
100% tests passed
```

### 4.2 单 GPU GEMM bench (warmup 10, iters 100, hipEvent 计时)

| M | N | K | per-iter (ms) | TFLOPS | % of MI355X BF16 peak (1300) |
|---|---|---|---|---|---|
| 256  | 14336 | 7168 | 0.0803 | 655  | 50% |
| 256  | 7168  | 2048 | 0.0245 | 306  | 24% |
| 4096 | 7168  | 2048 | 0.1143 | 1052 | 81% |
| 4096 | 4096  | 7168 | 0.1976 | **1216** | **94%** |
| **8192** | **4096**  | **7168** | **0.3730** | **1290** | **99%** |
| 4096 | 14336 | 7168 | 0.7032 | 1197 | 92% |

`8192×4096×7168` 几乎贴 BF16 peak (99%)，证明 `mfma_tile.h` 的 hot loop 在新仓库里**移植无损**。`4096×4096×7168` (DSv3 grouped GEMM 一个 expert 的典型 per-call shape) **1216 TFLOPS / GPU**，**远超 M0 验收门槛 950 TFLOPS**。

小 M (M=256) 路径 TFLOPS 掉到 24-50%，原因是只能起 ceil(M/128) × ceil(N/128) = 2×56 = 112 (256x14336) 或 2×16 = 32 (256x7168) 个 WG，MI355X 256 CU 没填满。这正是 MonolithEP "small tile" 路径要特殊处理的地方，M3 / M7 会带回来。

### 4.3 vs PyTorch baseline (defer)

按 skill rule 1, 每次有 e2e 数字应当跟 PyTorch+RCCL baseline pair 报。M0 还没 e2e (只有单 GPU GEMM)，所以这次只起了 baseline 骨架 `baselines/pt_rccl_moe.py`，等 M3 super-kernel 5-phase 跑通后再启用对比。这点已经写在 skill 文档里。

## 5. 解释

- **为什么 GEMM 直接打到 99% peak**: `mfma_tile.h` 已经把 (XOR swizzle + DTOLDS + N_LDS_BUFS=2 双 buffer + K-step 同步加载) 这一套在 MonolithEP 调了 6 个月，单 tile 路径几乎没空间再优化。M0 的 wrapper 只是“调函数”，所以能继承 99% util。
- **column 偏移 bug 暴露的设计契约**: `mfma_tile.h` 的输出 c_out 期望 caller 已经把 (row, col) 都偏移好，自己只关心 within-tile 坐标 + n_off (n_off 仅给 B-tile 选 N 行用)。这种约定在 super-kernel 里 caller 一直是同一个（`expert_compute_phase`），所以契约自然清晰；当我把 `gemm.hip` 拆成纯 standalone GEMM wrapper 时这个约定容易丢失。RocMoE-v2 后续 phase 在 caller 一定要严格守这个约定。

## 6. 下一步 (M1)

按设计文档 §5 + skill milestone 表，**M1 = Layout-P + 64-bit `block_ready` bitmap + receiver-pull dispatch**：

1. 引入 `csrc/include/rocmoe/{workspace.h, types.h}` (扩展 types.h 加 MoEConfig)
2. 引入 `dispatch_body.h`，搬 RocMoE-bak 的 receiver-pull body, 改成按 (expert, block_b) 拉
3. 加 `block_ready[expert][block_b]` 64-bit bitmap，sender release / receiver acquire
4. `dispatch.hip` + launcher
5. ctest: `test_dispatch` 验证 receive 排布 + bitmap 状态
6. 单段 dispatch wall ≤ 1 ms / 8 GPU @ T_src=2048 验收

预期 flag: `UP` (vs MonolithEP 的 push 路径 dispatch 部分应该有 ~10% 改善, 主要在没有 g=0 fan-in spin)。

## 7. 相关文件

- 设计原始 note: [`2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md)
- skill: `~/workspace/slab/.cursor/skills/rocmoe-dev-loop/SKILL.md` (跨 session workflow)
- 同款架构 reference: `~/workspace/RocMoE-bak/csrc/include/rocmoe/{lds_layout,mfma_tile}.h`
- GEMM hot loop reference: `~/workspace/MonolithEP/csrc/include/monolith/mfma_tile.h`
