# GPT-OSS-20B / MI355X — `turbo_sync_free_moe_stage` 全 stage 实测：仅 stage 1 +0.5%，stage 2/3 反慢

**日期**：2026-04-25
**仓库**：`/home/xiaompen/mlperf-training/small_llm_moe_pretraining/primus`
**前情**：note 20 / 21 已经把 `use_turbo_grouped_mlp:true` 在 EP=1 反慢 -7% 的
原因锁到 stream-level（grouped GEMM 折叠成 stream 0 的 persistent kernel，
4-lane 副 stream 退化为 1-lane）。本 note 是把 `turbo_sync_free_moe_stage`
四个 stage 全跑了 trace，把官方文档"stage 2 is recommended for better
performance"这句在我们这套配置下的 trap 翻译清楚。

## 1. 触发动机

`Primus/docs/backends/megatron/patch-notes.md:49` 推荐 `stage 2`，前置条件
`enable_primus_turbo=True` 我们已经满足。直觉上 sync-free 砍掉 host-side
GPU 同步点，每 MoE 层都能省掉一个 `cudaStreamSynchronize`，估计 +2~5%。
所以挨个 stage 试，确认效果。

## 2. 关键源码：stage → auto-enable 映射（必读）

`Primus/primus/modules/trainer/megatron/utils.py:451`：

```python
def _get_sync_free_moe_options(stage: int) -> dict:
    sync_free_moe = {
        1: {"moe_use_fused_router_with_aux_score": True,
            "moe_permute_fusion": True},
        2: {"moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_mlp": True},                 # ← 强制 override
        3: {"moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_mlp": True,                  # ← 同样 force
            "use_turbo_fused_act_with_probs": True},
    }
    return sync_free_moe[stage]
```

`utils.py:519` 通过 `setattr(args, flag, value)` **直接覆盖 yaml**，所以即便
yaml 写 `use_turbo_grouped_mlp: false`，stage>=2 也会被翻成 True。

→ 这意味着 stage 2/3 自动踩进 note 21 已经证明的 -7% 坑。stage 1 是唯一一档
不动 grouped MLP 路径的、对 EP=1 安全的开关。

## 3. 实验设置（自包含可复现）

```
small_llm_moe_pretraining/primus/run-trace/20260425_b_v1fused/
  A_fuse0/        baseline                                       (note 19)
  B_fuse1/        + V1 fused residual                            (note 19)
  C_turbo_grouped/+ use_turbo_grouped_mlp:true                   (note 21)
  D_moe_overlap/  + patch_moe_overlap:true                       (note 21 §6 ①)
  E_sync_free_s2/ + turbo_sync_free_moe_stage:2  (本 note)
  F_sync_free_s3/ + turbo_sync_free_moe_stage:3  (本 note)
  G_sync_free_s1/ + turbo_sync_free_moe_stage:1  (本 note)        ← 唯一 +0.5%
```

24 train iter，profile steps 16..19 on rank 2，GBS=32 MBS=4，TP1 PP1 EP1，
seq=8192，FP8 hybrid。每个目录下 `run.sh` + `*.yaml` + `*.sh` 配齐，
单一 `bash run.sh` 即可重现。

## 4. 标量 + ProfilerStep 实测

### 4.1 iter 20 训练实测

| | B (baseline) | E (stage 2) | F (stage 3) | G (stage 1) |
|---|---|---|---|---|
| iter 10 ms | 1934.0 | 1947.9 | 1917.6 | **1897.6** |
| iter 20 ms | **1186.5** | 1269.4 | 1243.1 | **1181.1** |
| iter 20 TFLOPS | **696.0** | 650.6 | 664.3 | **699.2** |
| Δ vs B | — | **+7.0%** | **+4.8%** | **-0.5%（首个正向）** |
| HIP mem | 244 GiB | 250 GiB (+6) | 243 GiB | 244 GiB |

### 4.2 ProfilerStep#17 wall（trace 内更精确）

| | B | E (s2) | F (s3) | G (s1) |
|---|---|---|---|---|
| step wall | **1125.12 ms** | 1224.09 ms | 1202.85 ms | **1119.70 ms** |
| Δ vs B | — | **+98.97 ms** | +77.73 ms | **-5.42 ms** |

### 4.3 stream 0 占用率（最关键的一行）

| | B | E (s2) | F (s3) | C (turbo grouped) | G (s1) |
|---|---|---|---|---|---|
| stream 0 busy | 600.24 (53.3%) | **1172.38 (95.8%)** | **1149.53 (95.6%)** | 1181.43 (95.9%) | **594.17 (53.1%)** |
| 4 副 stream gemm | 822 ms 4-lane | **0** | **0** | **0** | **828 ms 4-lane ✓** |

→ E/F 跟 C 完全同一份指纹（stream 0 95~96%、4 副 stream 全消失），
G 完全保留 baseline B 的 4-lane 拓扑。

### 4.4 grouped_gemm 是不是被 force 进 turbo 路径

| | B | E (s2) | F (s3) | G (s1) |
|---|---|---|---|---|
| `_grouped_bf16_persistent_gemm_kernel` (top1) | — | **373.26 ms (30.5%)** | **373.32 ms (31.0%)** | **不出现** |
| `_grouped_variable_k_gemm_kernel` | — | 193.23 ms | 192.94 ms | 不出现 |
| `Cijk_*` (legacy GEMM 集合) | 947.93 | 127.06 | 127.17 | **956.50 ✓** |

→ E/F 把 32 expert 折叠成单个 persistent kernel 强行排到 stream 0；
G 维持 N×cuBLAS GEMM 散到 4 副 stream 的并行计算。

## 5. G (stage 1) 的净改动到底是什么？

stage 1 banner 实测：

```
========== Enable Sync-Free MoE Stage 1 (Auto-Enabled Options) ==========
moe_use_fused_router_with_aux_score..................................True
moe_permute_fusion...................................................True
========== Enable Sync-Free MoE Stage 1 (Auto-Enabled Options) ==========
```

我们 yaml 实际状态：
- `moe_permute_fusion: True`（**本来就开**，stage 1 设它属于幂等）
- `moe_use_fused_router_with_aux_score`：默认 `False` → 被 stage 1 翻成 `True`

→ G 相对 B 的唯一净 delta = `moe_use_fused_router_with_aux_score` 从 False
变 True。

这正对应 patch-notes.md 第 45 行：
> Fused router topk and calculation of moe aux loss score. Need Primus turbo
> backend. Used to **reduce launch overhead of the small kernels in router**.

trace 上确实看见 router 段 small kernel 数量减少（B 5,743 GPU kernels →
G 5,983 个但每个更短，per-step wall 净 -5.4 ms），符合 fused router 把
"topk + aux score 计算 + load balancing"砍成更少的 fused kernel 的预期。

## 6. AccumulateGrad 警告：所有 stage>0 都有，但只有 stage 2/3 真伤性能

E/F/G 都报 6 个 rank 的：

```
UserWarning: The AccumulateGrad node's stream does not match the stream of
the node that produced the incoming gradient. This may incur unnecessary
synchronization and break CUDA graph capture if the AccumulateGrad node's
stream is the default stream.
```

→ sync-free MoE 重写了 autograd stream 拓扑，所有 stage 都触发。但 G 的
trace 显示 stream 拓扑跟 B 完全一致（stream 0 仍 53%、4 副 stream gemm
正常），所以这条警告对 G 是 cosmetic，对 E/F 是 cosmetic + 已被
`use_turbo_grouped_mlp` override 的 -7% 主因吃掉了。

## 7. 决定 & yaml 修改

主仓库 yaml 已切到 stage 1：

```yaml
small_llm_moe_pretraining/primus/gpt_oss_20B-pretrain-fp8.yaml:194-211
turbo_sync_free_moe_stage: 1   # was 0; +0.5% TFLOPS confirmed
```

并把上面的判据（stage 2/3 force `use_turbo_grouped_mlp:True` → 跟 EP=1
4-lane 拓扑冲突 → -5~7%）写进 inline 注释，避免后续被"官方文档说 stage 2
最好"的直觉再带偏。

## 8. 与其他 patch-notes 参数的关系（评估清单）

| 参数 | 当前 | 评估 | 决定 |
|---|---|---|---|
| `turbo_sync_free_moe_stage` | 0→**1** | 实测 -5.4 ms | ✅ 切 1 |
| `use_turbo_deepep` | false | stage 2/3 顺带开了，但需要 `moe_token_dispatcher_type=flex` 才生效；当前用 `alltoall` | 跳过，先验证 dispatcher 切换的连锁影响 |
| `moe_use_fused_router_with_aux_score` | false→True (via stage 1) | 已被 stage 1 间接打开 | ✅ 已生效 |
| `moe_permute_fusion` | true | 已开 | — |
| `use_turbo_grouped_mlp` | false | note 21 已证 -7% in EP=1 | ⛔ 锁 false |
| `patch_moe_overlap` | false | note 21 §6 已证 EP=1 无效 | ⛔ 锁 false |
| `use_turbo_attention` | false | 未试，attn_kernel 200 ms 是 critical path 项，下次单独评估 | 待办 |
| `use_turbo_fused_act_with_probs` | false (stage 3 才开) | stage 3 -4.6% 里它的贡献被 grouped_mlp 反向淹没了，无法单独定量 | 待办：手动 setattr 后单独跑 |
| `no_fp8_weight_transpose_cache` | false | 内存优化 flag，省 mem 但可能伤 perf；目前 244 GiB 不紧张 | 跳过 |
| `disable_last_saving` | false | 跑 24 iter ckpt save 占 6 min，profile 用例没必要 | 待办：set true 节省每次 trace 的 6 min |

## 9. 工件位置

- B trace + breakdown：`small_llm_moe_pretraining/primus/run-trace/20260425_b_v1fused/B_fuse1/`
- E trace + breakdown：`.../E_sync_free_s2/`
- F trace + breakdown：`.../F_sync_free_s3/`
- G trace + breakdown：`.../G_sync_free_s1/` ← 新基线

复现：`bash <dir>/run.sh`，~6 min 训练 + ~6 min ckpt save。每跑完一个 leg
docker exec rm 一下 `<dir>/output/.../checkpoints` 释放 268 GB。

## 10. 下一步候选（按 ROI 重排）

1. **`use_turbo_attention: true`**：attn_kernel 占 ~200 ms (17% of step)
   是单点最大头，turbo 路径用 fmha 更好的 schedule，可能 -10~30 ms。
   单变量试。
2. **手动 `use_turbo_fused_act_with_probs: true` + 保持 stage 1**：避开
   grouped_mlp 反向，单独评估 fused act 的真实增量。可能 -2~5 ms。
3. **disk hygiene**：在 yaml 里把 `disable_last_saving` 翻 true，每次 trace
   节省 6 min ckpt + 268 GB 磁盘。
4. **V2 fused residual (`PRIMUS_FUSED_RESIDUAL_NORM_V2=1`)**：吃掉 ADD#2，
   note 18 估 -5~10 ms。
5. dtype/view tax audit（note 16 §下一波）：`direct_copy 22 ms +
   manual_unroll 22 ms + AUnaryFunctor 23 ms` → -10~25 ms。
