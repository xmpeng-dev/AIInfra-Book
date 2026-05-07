# mlperf-llama — Llama-2-70B LoRA SFT, NeMo vs Primus (MI355X)

> **目标**：在 8×MI355X 上把 Primus 的 Llama-2-70B LoRA SFT 跑到 ≥ NeMo 的 step time 与可控 HBM
> **平台**：1 节点 × 8 MI355X (288 GiB HBM, gfx950)
> **栈**：`bf16_with_fp8_hybrid`（FP8 GEMM + CK V3 attention）, GBS=8 / MBS=1 / seq=8192 packed, TP=PP=CP=1, DP=8
> **镜像**：NeMo `rocm/amd-mlperf:llama2_70b_sft_nemo_6.0` · Primus `rocm/mlperf-training:llama2_70b_training_6.0_2026-04-27-22-49-59`

## 状态

| 维度 | NeMo | Primus | 差距 |
|---|---:|---:|---:|
| Step time (ProfilerStep) | **1490 ms** | **1626 ms** | NeMo 快 **8.4 %** |
| Compute-stream busy | 99.5 % | 84.9 % | +14.6 pp |
| Idle (DataLoader) | 7 ms / 0.5 % | **191 ms / 11.8 %** | **−184 ms** |
| RCCL on critical path | 5 ms (LoRA-A2A) | 54 ms (DDP RS 暴露) | −49 ms |
| Peak VRAM | ~200 GB | **285.84 GB** | NeMo 省 ~85 GB |
| 收敛 | **0.9244 @ step 384, 10.79 min** ✓ | 未跑到 target | NeMo 端到端达标 |

**一句话结论**：perf gap 几乎全部来自 Primus 单进程 DataLoader（−191 ms idle）；HBM gap ~70/85 GB 来自 `fp8_param=True`（FP8 weight 1B + bf16 master 2B = 3B/param），由 bridge precision recipe 在运行时强制覆盖，**yaml 看不到**。

项目状态：**active**，2026-04-29 启动。

## 进展时间线

| 日期 | 里程碑 | 关键数字 | 来源 note |
|---|---|---|---|
| 2026-04-29 | NeMo 单边 torch profiler trace 拆解 | step 1.502 s · 5.40 sps · TTT 647 s · FP8 GEMM 51% / Attention 21% / NCCL <2% | [`nemo_llama2_70b_lora_torchprof`](./2026-04-29_nemo_llama2_70b_lora_torchprof.md) |
| 2026-04-29 | Primus baseline (DDP) trace 拆解 | step 1.62 s · 2251 TFLOP/s/GPU · FP8 GEMM 49.9% · NCCL 串行 (overlap 0.4%) · 191 ms 起始 idle · VRAM 95.6% reserved (TIGHT) | [`llama2_70b_lora_baseline_trace_breakdown`](./2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md) |
| 2026-04-29 | NeMo 详细配置档 | 完整 mcore TransformerConfig / DDP / Optimizer / MixedPrecision / LoRA / env / profiler | [`nemo_llama2_70b_lora_config`](./2026-04-29_nemo_llama2_70b_lora_config.md) |
| 2026-04-29 | NeMo vs Primus 配置 / 性能对比 | 5 个高 HBM-impact 配置项定位（fp8_param +70 GB 主因）+ 工作流差异说明 | [`nemo_vs_primus_config_diff`](./2026-04-29_nemo_vs_primus_config_diff.md) |
| 2026-04-29 | **191 ms idle 真因定位** | Layer 1：单进程 DataLoader + 主线程 collate（`aten::ones`/`tril`/`lt` 8192² mask 共 ~125 ms） | [`idle_191ms_layer1_dataloader_root_cause`](./2026-04-29_idle_191ms_layer1_dataloader_root_cause.md) |
| 2026-04-29 | DataLoader workers 修复尝试 | ❌ Fork-after-CUDA deadlock；❌ `return_cu_seqlen=True` 与 `fused_single_qkv_rope` 不兼容；revert 到 baseline；需 spawn context 改 Bridge `dataset_provider` | [`nemo_vs_primus_meeting_summary_en`](./2026-04-29_nemo_vs_primus_meeting_summary_en.md) |
| 2026-04-29 | Meeting deck (EN) + Cursor canvas walkthrough | 给 manager 用的 1.5–2 min talk track（speed gap + memory gap） | [`nemo_vs_primus_meeting_summary_en`](./2026-04-29_nemo_vs_primus_meeting_summary_en.md), [`nemo_vs_primus_canvas_walkthrough_en`](./2026-04-29_nemo_vs_primus_canvas_walkthrough_en.md), [`nemo-vs-primus-llama2-70b-lora-trace.html`](./nemo-vs-primus-llama2-70b-lora-trace.html) |

## 下一步（按 ROI）

| 优先级 | 方向 | 预期 | 备注 |
|---|---|---|---|
| P0 | **DataLoader prefetch 修复**：Bridge `dataset_provider` 接 `multiprocessing_context=spawn`，或 worker-side CUDA init | step −150~190 ms（Primus 应反超 NeMo） | 不是一行 yaml；要碰代码 |
| P0 | **A/B `fp8_param=True → False`**（patch Primus bridge precision recipe `_apply_precision_overrides`） | HBM −70 GB（验证主因） | 必须改 recipe，不能动 yaml |
| P1 | A/B `fp8_param_gather`、`grad_reduce_in_fp32`（同样在 bridge precision recipe） | HBM 各 −5~10 GB | |
| P1 | A/B 6 个 DDP overlap 旋钮（recipe `llama2_custom.py:585-594` 写死 True） | HBM −10~15 GB（bucket padding） | |
| P2 | `enable_primus_turbo + use_transformer_engine_op_fuser` HBM/time trade-off 量化 | NeMo 多 1.2 GB HtoD + 1193 ms transpose（已 overlap） | |
| P2 | LOGGING_INTERVAL 调大（NeMo 端已有 `log_every_n_steps=1` 高频 GPU↔host scalar sync） | NeMo step −30~50 ms | 见 NeMo trace 4.6 节 |

## 文件索引

| 主题 | 文件 |
|---|---|
| Trace 拆解 (NeMo) | [`2026-04-29_nemo_llama2_70b_lora_torchprof.md`](./2026-04-29_nemo_llama2_70b_lora_torchprof.md) |
| Trace 拆解 (Primus baseline) | [`2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md`](./2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md) |
| 配置档 | [`2026-04-29_nemo_llama2_70b_lora_config.md`](./2026-04-29_nemo_llama2_70b_lora_config.md), [`2026-04-29_nemo_vs_primus_config_diff.md`](./2026-04-29_nemo_vs_primus_config_diff.md) |
| 真因定位 | [`2026-04-29_idle_191ms_layer1_dataloader_root_cause.md`](./2026-04-29_idle_191ms_layer1_dataloader_root_cause.md) |
| 对外材料 | [`2026-04-29_nemo_vs_primus_meeting_summary_en.md`](./2026-04-29_nemo_vs_primus_meeting_summary_en.md), [`2026-04-29_nemo_vs_primus_canvas_walkthrough_en.md`](./2026-04-29_nemo_vs_primus_canvas_walkthrough_en.md), [`nemo-vs-primus-llama2-70b-lora-trace.html`](./nemo-vs-primus-llama2-70b-lora-trace.html) |

## 维护约定

每写一篇 note → 同时回写本 README 的 **进展时间线** 和 **下一步**。
