# Mega MoE MXFP8 — 通信机制 epoch 化 + 全 stage 性能快照（T=8192）@ MI355X

> **用途**: 记录 fp8(MXFP8) mega MoE 的**通信机制从 host-reset 迁到 bf16 epoch 自复位**（2026-07-23），以及迁移后**前向 + 反向全 stage** 的 fp8-vs-bf16 性能快照。
> **Where**: `smci355-ccs-aus-n01-21`（MI355X / gfx950 ×8），容器 `xiaoming-dev`。job 22391。
> **配置**: DeepSeek-V3，EP8，**T=8192**，H=7168，I=2048，E=256，topk=8，BM=BN=256，routing=load_balanced。基线 = 本仓库 bf16 `mega_moe_fused` 同族 FlyDSL bf16 kernel。跨 8 rank 取最慢；fp8 精度 SNR(dB)=`10·log10(‖ref‖²/‖ref−out‖²)`。

## 本次改动（通信机制 epoch 自复位）

把整条 fp8 dispatch/combine 的**跨 rank 同步协议**从 host-reset（每次调用 `torch.cuda.synchronize()+group.barrier()+scoreboard/sb_l2/barrier_local 清零`）换成 **bf16 式设备端 epoch 自复位**：

- **双 bank i64 flag**（parity 选 bank）+ 设备端 `epoch_bump` kernel（翻 parity、累加 per-bank expected）+ spin 到累积 expected，**不消费复位**。
- signal heap：`scoreboard/sb_l2/barrier_local` → `dispatch_flag / preshuffle_flag / combine_flag / reduce_flag`（都 i64、2 bank）；删 `sb_consume`。
- dispatch comm signal 从 per-pool-block 变长 → **per-expert uniform**（`dispatch_flag[bank+expert]+=1`，恒 `num_ranks`），与 bf16 一致。
- 删掉 dispatch/combine wrapper 的 `_host_rendezvous`、prologue 的 in-kernel scoreboard/barrier_local 复位、bwd op 层的 host flag reset。

**收益**：(1) 前向/反向 compute 路径**彻底无 host `synchronize()+barrier()`**；(2) 消除了大 T 的 **reset 竞争死锁**（原 STEP3 在 T=8192 会挂 —— 根因是 host 清零与 in-flight kernel/peer 抢，不是 tail-reduce 共调度）。

> ⚠️ **CUDA-graph 仍不支持**：host sync 去掉了,但跨 rank symm-memory PUSH + 设备端 spin-wait 握手与 graph capture 不兼容(各 rank 独立 capture 会互等 → 挂)。custom_op 也不适用(bf16 mega MoE 同样是 autograd.Function：schema 装不下 group/handle/live-symm)。

同期清理:`gemm_mxfp8_tile.py` 删死代码(测试壳 + raw/LDS scale 路径),784→297 行,核心 tile 只留 preshuffled。

## 复现

```bash
cd /perf_apps/xiaoming/MegaMoE
# 逐 stage 跑,每 stage 前清残留进程 + 等 VRAM 回 baseline(推荐):
srun --jobid <J> --overlap docker exec xiaoming-dev bash \
  benchmark/ops/training/run_stages_fp8.sh            # env: T MODE WARMUP ITERS TIMEOUT STAGES
# e2e fwd+bwd + 梯度 SNR:
PYTHONPATH=$PWD python benchmark/ops/bench_mega_moe_fused_fp8_bwd.py --num-processes 8 --num-tokens 8192
# 全前向 SNR gate:
PYTHONPATH=$PWD python benchmark/ops/bench_mega_moe_fused_fp8.py --num-processes 8 --num-tokens 8192
```

## 当前快照（T=8192, load_balanced, clean-env runner, 2026-07-23）

| 阶段 | fp8 (ms / TFLOPS) | bf16 (ms / TFLOPS) | bf16/fp8 | 精度 | 状态 |
|---|---|---|---|---|---|
| **L1** dispatch+fc1 | 2.455 / 1665 | 4.146 / 986 | **1.69×** | cos=1.0 | ✅ |
| **L2** fc2+combine | 2.094 / 976 | 3.027 / 675 | **1.45×** | finite | ✅ |
| **FWD** 全前向 (L1+SwiGLU+L2) | 4.809 | 6.840 | **1.42×** | SNR 22.3 dB | ✅ |
| **STEP1** dispatch(dy)+fc2-dgrad | 1.678 / 1218 | 2.471 / 828 | **1.47×** | SNR 30.9 dB | ✅ |
| **dW2** fc2 wgrad (variable-K) | 1.797 / 1138 (FULL) | 2.092 / 977 | FULL **1.16×** / GEMM-only **2.0×** (1957 TFLOPS) | SNR 21.8 dB | ✅ |
| **dW1** fc1 wgrad (variable-K, LOCAL) | 2.931 / 1395 (FULL) | 4.048 / 1010¹ | FULL **1.38×** / GEMM-only **2.05×** (2068 TFLOPS) | SNR 21.8 dB | ✅ |
| **STEP3** fc1 dgrad+combine | 3.15 / 1298 | ~3.9² | **~1.24×** | dx finite (norm 9.9e2) | ✅ fp8 稳² |
| **e2e** fwd+bwd (完整 autograd) | 14.87 ms | 19.94 ms | **1.34×** | dx21.9/dtw23.1/dW1 19.5/dW2 19.7 dB PASS | ✅（含 STEP3, T=8192 无 hang）|

- ¹ dW1 的 bf16 参考腿只算 GEMM（同一本地 pool）；fp8 dW1 是 LOCAL（复用前向派发的 fc1-input 池），还省掉 bf16 融合路径对 `saved_x` 的跨 rank 重派发 → 实际优势 > 1.38×。
- ² **fp8 STEP3 在 T=8192 完全正常**：standalone 3.15 ms / 1298 TFLOPS，dx finite；e2e bwd T=8192 dx SNR 21.9 dB、无 hang。之前看到的 T=8192 SIGABRT **是 bench 里的 bf16 *参考腿* 崩,不是 fp8**——用 `[STEP3 fp8]` 先打印证实 fp8 leg 已完成(fin=True)后才 abort。bf16 参考 combine 在 **K=2I=4096 + T=8192** 会 OOB(nt/nn 都崩;profile_l2 的 K=I=2048 bf16 combine T=8192 正常),是未测试的大-K bf16-combine 配置,**与 epoch 通信机制 / fp8 路径无关**。bf16 参考值 ~3.9 ms 取自 2026-07-21 note(独立 bf16 nt),故 ~1.24×。standalone STEP3 在 T=2048 两腿都干净:fp8 1.143 / bf16 1.371 = 1.20×。
- ³ 另:反复"清进程后仍崩"的调查中出现的 srun 秒 137,**是探针命令里 `pkill -f bench_mega_moe_fp8.py` 把 bash -c 自己杀了**(bash -c 字符串含 bench 名),不是 GPU reset;`run_stages_fp8.sh` 无此问题(自身 cmdline 是 .sh 路径)。

## 结论

- 通信机制 epoch 化后：整条 fp8 mega MoE(fwd+bwd)正确(fwd SNR 22 / 梯度 SNR 19.5–23)、compute 路径无 host sync、T=8192 e2e 稳定不挂,**所有 stage fp8 均快于 bf16(1.16–1.69×)**。
- 唯一遗留:STEP3 **standalone** bench 在 T=8192 崩(harness 问题,非 kernel/通信机制)。

## commits（feat/xiaompen/mega_moe_flydsl_mxfp8）

- `99f7c1cf` fwd 通信 → epoch 自复位
- `42c4b4d9` bwd 通信 → epoch 自复位
- `4b0efcc6` gemm_mxfp8_tile 死代码清理(raw/LDS scale + 测试壳)
