# Turbo MoE benchmarks

Standalone benchmarks for the Primus Turbo MoE kernels used by the
`examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml` pretrain
recipe.

## `bench_turbo_moe_e2e.py`

End-to-end baseline for one MoE layer in the **Turbo** flavor:

```
hidden ─► dispatch_preprocess ─► token_dispatch (DeepEP A2A) ─► dispatch_postprocess
       ─► grouped_gemm (fc1) ─► swiglu_with_probs ─► grouped_gemm (fc2)
       ─► combine_preprocess  ─► token_combine  (DeepEP A2A) ─► combine_postprocess
       ─► hidden
```

The pipeline matches what Megatron's `MoELayer` drives at runtime once the
following Primus patches/extensions take effect (all enabled by the yaml above
plus `enable_primus_turbo: true`):

- `primus/backends/megatron/patches/turbo/moe_dispatcher_patches.py` —
  swaps `megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher`
  for `PrimusTurboDeepEPTokenDispatcher`.
- `primus/backends/megatron/core/extensions/primus_turbo.py` —
  - `PrimusTurboDeepEPTokenDispatcher` splits dispatch/combine into the six
    fine-grained stages exposed above.
  - `PrimusTurboGroupedMLP` implements `fc1 = grouped_gemm`, `act = swiglu_with_probs`,
    `fc2 = grouped_gemm`. The benchmark uses the same kernel sequence so the
    numbers reflect the "fully turbo" MoE even though the yaml currently has
    `use_turbo_grouped_mlp: false` (TEGroupedMLP) and only the dispatcher/combine
    on the turbo path.

### Defaults

Pulled from `primus/configs/models/megatron/deepseek_v3.yaml` plus the MI355X
overrides:

| knob | default |
|---|---|
| hidden_size | 7168 |
| moe_ffn_hidden_size | 2048 |
| num_experts | 256 |
| topk | 8 |
| micro_batch_size | 2 |
| seq_length | 4096 |
| dtype | bf16 |
| EP | `world_size` (8 on a single MI355X node) |
| TP | 1 |
| `turbo_deepep_num_cu` | 80 |
| `turbo_deepep_use_comm_stream` | false |
| `moe_permute_fusion` | true |
| `turbo_sync_free_moe_stage` | 1 |
| `moe_router_force_load_balancing` | true |

### Running

```bash
# Single MI355X node, EP=8, full-shape DeepSeek-V3 MoE layer:
torchrun --standalone --nproc-per-node=8 \
    benchmark/kernel/moe/bench_turbo_moe_e2e.py

# Forward-only:
torchrun --standalone --nproc-per-node=8 \
    benchmark/kernel/moe/bench_turbo_moe_e2e.py --mode fwd

# Tiny smoke run on 2 GPUs:
torchrun --standalone --nproc-per-node=2 \
    benchmark/kernel/moe/bench_turbo_moe_e2e.py \
    --num-experts 16 --topk 4 --seq-length 1024 --micro-batch-size 1

# Write CSV for downstream analysis (rank-0 only):
torchrun --standalone --nproc-per-node=8 \
    benchmark/kernel/moe/bench_turbo_moe_e2e.py \
    --output-csv ./turbo_moe_e2e.csv
```

### Output

For each iteration the script records CUDA-event timings for every stage and
prints a per-rank table on rank 0:

```
[fwd_bwd] per-stage timing (CUDA events, this rank):
+-----------------------+-----------+-------------+----------+----------+----------+
| Stage                 | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) |
+-----------------------+-----------+-------------+----------+----------+----------+
| dispatch_preprocess   | ...
| token_dispatch        | ...
| dispatch_postprocess  | ...
| grouped_gemm_fc1      | ...
| swiglu_with_probs     | ...
| grouped_gemm_fc2      | ...
| combine_preprocess    | ...
| token_combine         | ...
| combine_postprocess   | ...
| [sum-of-stages]       | ...
+-----------------------+-----------+-------------+----------+----------+----------+

[e2e] forward  mean (ms, rank0): ...
[e2e] forward  mean (ms, avg over ranks): ...
[e2e] backward mean (ms, rank0): ...
[e2e] backward mean (ms, avg over ranks): ...

[fc] grouped_gemm fc1+fc2 mean (ms): ...  => ... TFLOP/s/rank
```
