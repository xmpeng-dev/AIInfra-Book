# constraints/config — Parallelism / Shape Legality

## Purpose

Pre-submit rules that reject **structurally invalid** plans —
parallelism shapes that the framework will reject at startup, or
that violate the model's own requirements. Cheap, fast, deterministic.

## Tool contract

```python
constraint.check(plan, cluster_profile) -> {
    valid: bool,
    violations: [
        {rule: <str>, observed: <val>, expected: <val>, hint: <str>},
        ...
    ],
}
```

`violations` is non-empty iff `valid = false`. Each violation cites
exactly one rule (mirrors `constraints/SKILL.md` § "One rule, one
rejection reason").

## Rule catalog

### World-size consistency

```yaml
- rule: world_size_match
  expr: tp × pp × dp == cluster_profile.nodes × gpus_per_node
  hint: "rebalance one of {tp, pp, dp} so the product equals total GPUs"

- rule: ep_divides_dp_or_world
  expr: ep divides dp     OR     ep divides world_size
  hint: "framework requirement; pick ep ∈ divisors(dp) or divisors(world_size)"
```

### Model dimension consistency

```yaml
- rule: tp_divides_num_heads
  expr: model.num_attention_heads % tp == 0
  hint: "pick tp ∈ divisors(num_heads); typical {1, 2, 4, 8}"

- rule: tp_divides_kv_heads_for_gqa
  expr: model.num_kv_heads % tp == 0     # GQA / MQA models
  hint: "for GQA, tp must also divide num_kv_heads"

- rule: pp_divides_num_layers
  expr: model.num_layers % pp == 0
  hint: "pick pp ∈ divisors(num_layers) for clean stage balance; otherwise framework may reject or auto-balance with imbalance"

- rule: ep_divides_num_experts
  expr: model.num_experts % ep == 0          # MoE only
  hint: "pick ep ∈ divisors(num_experts)"
```

### Microbatch consistency

```yaml
- rule: gbs_decomposes
  expr: gbs == mbs × dp × num_microbatch     # num_microbatch = gbs / (mbs × dp)
  hint: "set gbs to a multiple of mbs × dp"

- rule: num_microbatch_geq_pp
  expr: num_microbatch >= pp                  # avoid trivial bubble
  hint: "with pp=P, need at least P microbatches; prefer ≥ 2P"

- rule: vpp_requires_pp_gt_1
  expr: vpp > 1 → pp > 1
  hint: "vpp interleaving only valid when pp > 1"
```

### Sequence-length / context-parallel

```yaml
- rule: cp_divides_seq_len
  expr: seq_len % cp == 0                     # if cp > 1
  hint: "context parallel requires seq_len divisible by cp"
```

### MoE-specific

```yaml
- rule: ep_le_world_size
  expr: ep <= world_size
- rule: top_k_le_num_experts
  expr: top_k <= num_experts
- rule: capacity_factor_in_range
  expr: 1.0 <= capacity_factor <= 2.0
  hint: "values >2 wastefully reserve memory; <1 incurs aggressive drops"
```

### Dtype consistency

```yaml
- rule: dtype_supported_by_cluster
  expr: dtype in cluster_profile.compute.supported_dtypes
  hint: "fp8 requires hardware support; check cluster_profile.peak_tflops_fp8 != null"

- rule: dtype_supported_by_model
  expr: model.dtype_compatible(dtype)
  hint: "some models (e.g. with fixed-precision attention) reject fp8"
```

### Numerical stability

```yaml
- rule: optim_state_dtype_for_dtype
  expr: dtype == fp8 → optim_state_dtype in {fp32}
  hint: "fp8 forward requires fp32 master weights for stable updates"
```

## How rules are extended

Adding a new structural axis requires:

1. Add an axis row to `axis_taxonomy.md`.
2. Add corresponding rules here so Re-Plan rejects illegal values
   pre-submit.
3. If the axis interacts with env (e.g. `cp` requires
   `NCCL_*` topology hint), add a rule to `constraints/env.md` too.

## Output ordering

When multiple rules fire, return them in this order (stable):

1. World-size consistency (gating: nothing else makes sense if this
   fails).
2. Model dimension consistency.
3. Microbatch consistency.
4. MoE-specific.
5. Dtype / numerical.
6. Sequence-length.

Re-Plan typically only relaxes the first violation, so ordering
controls what gets re-tried first.

## Hard vs soft

All rules above are **hard** (reject pre-submit). For soft warnings
(e.g. unusual but legal `mbs / gbs` ratios) emit a `warning` instead
of a `violation`; submit proceeds, but Settle may down-weight.

```yaml
- warning: very_small_num_microbatch
  expr: pp > 1 AND num_microbatch < 2 × pp
  hint: "bubble will dominate; consider increasing num_microbatch (mbs↓ or gbs↑)"
```

## Cross-links

- Axis registry → `workflow/axis_taxonomy.md`
- Env compatibility → `constraints/env.md`
- Memory feasibility → `constraints/oom.md`
- Failure-side mapping → `constraints/validation.md`

## Scale-aware notes

| Family | Single node | Multi node |
|--------|-------------|-----------|
| World-size rule | trivially `tp × pp × dp == gpus_per_node` | full `tp × pp × dp == nodes × gpus_per_node` |
| Stage balance (`pp_divides_num_layers`) | rare (`pp = 1`) | central |
| `cp` rules | rarely used | used for long-context training |
| Validation cost | < 1 ms | < 1 ms |
