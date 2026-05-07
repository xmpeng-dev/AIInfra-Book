# NeMo Llama2-70B LoRA SFT — Torch Profiler Trace 分析

- **日期**：2026-04-29
- **平台**：1× MI355X（单节点 8 GPU）
- **镜像**：`rocm/mlperf-training:llama2_70b_training_6.0_2026-04-27-22-49-59`
- **代码**：`llama2_sft/nemo`
- **本次 trace 目录**：
  `~/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_054719/`
- **启动命令**：
  ```bash
  TORCH_PROFILE=1 bash run.sh   # CONFIG_FILE 默认 config_MI355X_1x8x1.sh
  ```
  （改造细节见 `run.sh` + `src/callbacks/custom_callbacks.py` 中的 `torch_prof` 分支）

---

## 1. 关键超参（取自 `hparams_summary.txt` / `env_after_config.txt`）

| 维度 | 值 |
|---|---|
| Topology | DGXNNODES=1, DGXNGPU=8, TP=PP=CP=1, SP=False |
| Batch | MBS=1, MINIBS=1, GBS=8, gradient_accumulation_steps=1 |
| LR / Sched | LR=4e-4, warmup_steps=0, max_steps=1024 |
| Precision | **FP8 hybrid**（`FP8=True`, `FP8_AMAX_ALGO=most_recent`, `FP8_AMAX_HISTORY=4`）；FP4=False |
| LoRA | rank=16, alpha=32, dropout=0.1, target=`linear_proj/linear_qkv` |
| Recompute | 全关（baseline FP8 跑无需激活重算） |
| Comm overlap | TP_COMM_OVERLAP=False；NCCL_MIN_P2P/CTAS=32 |
| Healing / FP4 | 关闭（FP4 路径在 `config_MI355X_1x8x1_fp4.sh`，未在本次跑） |

> 完整解析后的 hydra cfg 在 `hydra_resolved/config.yaml`（软链到
> `outputs/2026-04-29/05-47-29/.hydra/config.yaml`）。

---

## 2. 总体时序（来自 :::MLLOG）

| 阶段 | 时间 | 说明 |
|---|---|---|
| `cache_clear` / `init_start` | t=0 | 进程起步 |
| `before_model_init` | +9.68 s | tokenizer + cfg 解析 |
| `after_model_init` | +0.31 s | NeMo 模型 + LoRA + ckpt restore |
| `eval_accuracy=3.629` (warmup eval) | +77 s | sanity eval（高 loss 是预期，LoRA 还没训） |
| `warmup_time` | 107.4 s | 自定义 warmup（编译 + Lightning val_loop 预 init） |
| `init_stop` / `run_start` | **117.5 s** | TTT 计时起点 |
| `block_start … block_stop … eval … RUN_STOP` | **+647.3 s** | 5 个 eval block，384 个 train step |
| **wall-clock total** | **793 s ≈ 13.2 min** | 含 init+warmup+train+eval |

### 收敛轨迹

| step | samples_count | eval_accuracy |
|---:|---:|---:|
| 0 | 0 | 3.6290 *(sanity)* |
| 192 | 1536 | **0.9415** ← 第一次 eval 已超 target 0.925 |
| 240 | 1920 | 0.9361 |
| 288 | 2304 | 0.9302 |
| 336 | 2688 | 0.9315 |
| 384 | 3072 | 0.9244 → **RUN_STOP success** |

### 稳态训练性能

| 指标 | 第 1 个 block | 第 2–5 个 block |
|---|---:|---:|
| `throughput` (samples/s) | 5.334 | 5.400 / 5.402 / 5.401 / 5.401 |
| `train_step_time` (s/step) | 1.524 | ~1.502 ± 0.001 |
| `validation_time` (s/eval) | 13.32 | 13.27 / 13.21 / 13.31 / 13.22 |

> 第一个 block 略慢 50 ms/step，符合 hipBLASLt / aiter JIT 余热的特征。**之后稳态非常平**（±0.5 ms）。

---

## 3. Profiler 配置

| 项 | 值 |
|---|---|
| 后端 | `torch.profiler` (`activities=[CPU,CUDA]`) |
| Schedule | `skip_first=1, wait=0, warmup=3, active=2, repeat=1` → 在 step 6 触发一次 `on_trace_ready`，之后 callback 主动 `stop()` |
| 输出 | `torchprof/key_avg_6_<uuid>.txt` ×8 + `trace_6_<uuid>.json` ×8 |
| 总大小 | ~345 MB（trace JSON 占 ~360 MB / 8 = 45 MB/卡） |
| 每 rank 一份 | 是（uuid 命名，8 个 md5 各异，符合预期） |
| 接入点 | `CustomCallback.on_train_batch_start/_end/on_train_end`（rpd 路径并列存在） |

> 已通过 active 2 步窗口结束后 `prof.stop()` 兜底，**对训练 wall-time 无后续开销**。
> 实测：profile 窗口 GPU 时间 = 2 × 1.481 s ≈ 2.96 s，与 MLLOG `train_step_time≈1.50 s` 一致 → profiler overhead 可忽略。

---

## 4. 算子热点（rank 0，2 个 active step 窗口）

> Self CUDA total = **7.525 s**；Self CPU total = 3.340 s
> 下面只列 self CUDA 占比 ≥1% 的项，类别已合并。

### 4.1 计算大头（FP8 GEMM）

| 类别 | Self CUDA | 占比 | calls | avg/call |
|---|---:|---:|---:|---:|
| `Custom_Cijk_Alik_Bljk_F8B8BS_…`（hipBLASLt FP8 GEMM, A=F8 B=F8） | 689.8 ms | 9.17 % | 480 | 1.437 ms |
| `Custom_Cijk_Alik_Bljk_F8BS_…`（hipBLASLt FP8 GEMM, single-side F8） | 543.7 ms | 7.23 % | 480 | 1.133 ms |
| `Custom_Cijk_Alik_Bljk_F8BS_…`（其它形状）  | 199.2 ms | 2.65 % | 160 | 1.245 ms |
| `Custom_Cijk_Alik_Bljk_F8B8BS_…`（其它形状） | 75.7 ms | 1.01 % | 158 | 0.479 ms |
| `aten::mm`（fallback BF16 mm，主要用在 LoRA-A/B 小矩阵） | 77.7 ms | 1.03 % | 1924 | 40 µs |
| **小计** | **~1.59 s** | **~21 %** | | |

→ FP8 LayerNorm-Linear 已经走通 hipBLASLt，4 类 kernel 共 480+480+160+158 ≈ **1.3k GEMM calls / 2 step**，平均 1.0–1.4 ms/call，是当前核心算力来源。

### 4.2 NeMo / TE 高层 module（含其内部多个 kernel）

| 模块 | Self CUDA | 占比 | calls |
|---|---:|---:|---:|
| `_LayerNormLinearBackward` | 640.5 ms | 8.51 % | 318 |
| `_LayerNormLinear` (FWD) | 509.4 ms | 6.77 % | 320 |
| `FusedAttnFuncBackward` | 463.4 ms | 6.16 % | 160 |
| `_LinearBackward` | 304.7 ms | 4.05 % | 320 |
| `_Linear` (FWD) | 288.6 ms | 3.84 % | 320 |
| `FusedAttnFunc` (FWD) | 177.4 ms | 2.36 % | 160 |

### 4.3 Attention（aiter / CK 路径）

| Kernel | Self CUDA | 占比 | calls |
|---|---:|---:|---:|
| `aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv` | 414.6 ms | 5.51 % | 160 |
| `aiter::fmha_fwd_hd128_bf16_causal` | 176.8 ms | 2.35 % | 160 |
| `ck_fused_attn::dk_dv_reduce` | 16.7 ms | 0.22 % | 160 |
| `aiter::fmha_bwd_hd128_dq_shuffle` | 7.8 ms | 0.10 % | 160 |
| **小计** | **~616 ms** | **~8.2 %** | |

> Attention 走 **aiter + CK**，bwd ≈ 2.34× fwd（合理，bwd 多一遍 dq/dk/dv）。

### 4.4 通信

| Kernel | Self CUDA | 占比 | calls | avg |
|---|---:|---:|---:|---:|
| `nccl:all_reduce` | 143.4 ms | 1.91 % | 6 | 23.9 ms |
| `nccl:all_gather_into_tensor_coalesced` | 4.0 ms | 0.05 % | 2 | 1.98 ms |
| `nccl:reduce_scatter_tensor_coalesced` | 0.8 ms | 0.01 % | 2 | 0.40 ms |

→ TP=PP=1，只剩 DDP grad all-reduce，单卡通信占比 **<2 %**，不是瓶颈。

### 4.5 cast / transpose / activation / norm

| Kernel | Self CUDA | 占比 | calls |
|---|---:|---:|---:|
| `_cast_transpose_triton` | 77.7 ms | 1.03 % | 318 |
| `transpose_optimized_kernel` (FP8 e4m3) | 57.5 ms | 0.76 % | 638 |
| `rocm_cast_only_kernel` (bf16→fp8 ×2) | 51.7 ms | 0.68 % | 800 |
| `SwiGLU` FWD+BWD | 100.0 ms | 1.33 % | 320 |
| RMSNorm FWD+BWD (`triton`) | 46.7 ms | 0.62 % | 642 |
| `aiter::rope_fwd/bwd` | ~25.5 ms | 0.34 % | 640 |
| Dropout FWD+BWD | 41.9 ms | 0.56 % | 638 |

→ 这些"小料"加起来不到 5 %，FP8 cast/transpose 已经走 triton + 优化的 ROCm kernel，没明显异常。

### 4.6 异常 / 值得关注的项 ⚠️

| 项 | 值 | 看法 |
|---|---|---|
| **`Memcpy HtoD (Host → Device)`** | **2.326 s / 30.91 %**, **12 calls**, avg **193.8 ms/次** | 是这次 trace 里**单项 GPU 时间最高的事件**。但它占的是 HtoD 拷贝引擎的"忙时"，而不是计算流的 wall-time——拷贝可与 compute 重叠。**需要打开 `trace_6_*.json` 时间线确认它有没有阻塞 main stream**。可疑触发点：①checkpoint resume 之后部分张量懒加载到 GPU；② LoRA 适配器 / FP8 amax / scale 张量从 CPU 上传；③ profiler 自身的 metadata 拷贝。如果是 ① 或 ②，**应当只发生在前几个 step**——profile 落在 step 6，正好覆盖了这种"温热"窗口，有可能稳态根本就没这么多 HtoD。 |
| `aten::copy_` | 2.338 s / 31.07 %, 546 calls | 与上面的 HtoD 高度耦合（HtoD 由 `aten::copy_` 触发）。同样建议在时间线里看是不是阻塞调用栈。|
| `aten::mul` self CUDA | 2.293 s / 30.47 %, 646 calls | 数量级偏大，疑为 FP8 amax/scale 路径 + LoRA α 缩放叠加。可在时间线里 group-by name 查具体调用点。|
| `aten::_local_scalar_dense` CPU side | **63.5 % Self CPU / 2.12 s** | 高频 GPU→host scalar 同步（40 calls，53 ms/call）。常见来源：loss `.item()`、grad-norm、log_every_n_steps=1 时的指标读取。**值得 attempt 把 logging frequency 从 1 改 5–10 看 step time 变化**。|

---

## 5. 时间分配（基于 active 窗口的粗估）

```
2 × ProfilerStep ≈ 2.96 s GPU time（≈ 1.48 s/step，与 MLLOG 1.50 s/step 吻合）

  FP8 GEMM (hipBLASLt Custom_Cijk_*)        : ~1.51 s   (~51 % active GPU time)
  Attention (aiter::fmha + ck reduce)       : ~0.62 s   (~21 %)
  cast/transpose/SwiGLU/RMSNorm/RoPE/Dropout: ~0.27 s   (~9 %)
  NCCL all-reduce                           : ~0.14 s   (~5 %)
  H2D / aten::copy_ overhead (待时间线确认) : ~0.30–2.3 s（与 compute 大概率部分重叠）
  其它 wrappers / 小 kernel                 : ~0.40 s
```

> `aten::copy_` / `Memcpy HtoD` 表面占比超大，但因为是异步拷贝，**真实 wall-time 阻塞需要 chrome trace 的 timeline 视图来量化**。下面给了下一步动作。

---

## 6. 结论

1. **训练健康**。第 1 个 eval 就达到 0.9415（target 0.925），5 个 eval block 后 RUN_STOP success，TTT = **647 s / 10.79 min**，稳态 throughput 5.40 samples/s、step time 1.50 s。
2. **算力路径已经"现代化"**：LayerNormLinear 走 hipBLASLt FP8 GEMM、Attention 走 aiter+CK、Norm/RoPE/Cast 走 triton 与 ROCm 优化 kernel。FP8 hybrid 已经吃满。
3. **通信不是瓶颈**：单节点 TP/PP/CP=1，只剩 DDP grad allreduce，<2 %。
4. **可疑热点**：12 次共 2.3 s 的 HtoD 拷贝 + 高频 `aten::_local_scalar_dense`（CPU 同步）值得在 timeline 里追因。

---

## 7. 下一步动作（按 ROI 排序）

1. **打开 `trace_6_<uuid>.json`** 进 [perfetto.dev](https://ui.perfetto.dev/)，按 stream 看 HtoD/copy 是否真的阻塞 main stream。如果它们只发生在 step 6 这种 "刚开始训练" 的窗口，就把 profiler 挪到稳态再 sample：
   ```bash
   TORCH_PROFILE=1 PROF_WARMUP_STEPS=20 PROF_ACTIVE_STEPS=2 bash run.sh
   # 或者再加 skip_first 让 profiler 完全落在稳态
   ```
2. **把 `LOGGING_INTERVAL` 调大试一下**（`config_MI355X_1x8x1.sh` 里默认 5000，但 train.py 里 `log_every_n_steps=1`，外部 trainer 实际还是每步 logging）：
   ```bash
   # 临时验证
   TORCH_PROFILE=1 LOGGING_INTERVAL=20 bash run.sh
   # 然后看 train_step_time 是否进一步降到 ~1.45s
   ```
   如果有效，再考虑 patch `train.py:log_every_n_steps`。
3. **跑 `_fp4` 配置对比**：
   ```bash
   CONFIG_FILE=config_MI355X_1x8x1_fp4.sh TORCH_PROFILE=1 bash run.sh
   ```
   用同样的 trace 目录结构 + perfetto 看 mxfp4 + healing 路径里 GEMM kernel 名是否切换到 `…F4…` 系列；同时验证之前那次 `rank=4 exitcode=1` 的失败是否复现。
4. **抓 shape 信息**（一次性诊断用，开销大）：
   ```bash
   TORCH_PROFILE=1 TORCHPROF_RECORD_SHAPES=1 PROF_WARMUP_STEPS=10 bash run.sh
   ```
   可在 perfetto 里按 input shape 区分同名 GEMM 看是否落到非最优 kernel 形状。

---

## 8. 复现 / 归档

```bash
# 完整目录
cd ~/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_054719

# 重要文件
hparams_summary.txt           # 关键超参一眼看
scripts/                      # 当时的 run.sh + config + run_and_time.sh
hydra_resolved/config.yaml    # hydra 解析后完整 cfg（软链）
env_diff.txt                  # source config 前后的 ENV diff
run_and_time.log              # MLLOG 完整训练日志
torchprof/key_avg_6_*.txt     # 每卡算子热点表（本 note 数据来源）
torchprof/trace_6_*.json      # 每卡 chrome trace（拖到 perfetto.dev）
```

后续做对比时，直接：
```bash
diff -u <旧>/hparams_summary.txt <新>/hparams_summary.txt
diff -u <旧>/env_after_config.txt <新>/env_after_config.txt
diff -u <旧>/hydra_resolved/config.yaml <新>/hydra_resolved/config.yaml
```
即可定位两次 run 的差异点。
