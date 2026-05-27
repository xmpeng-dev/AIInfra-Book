###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Turbo MoE end-to-end baseline benchmark.

Reproduces, in a single file, the per-token pipeline of one MoE layer when the
DeepSeek-V3 example config is run with Primus Turbo enabled:

    examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml

The pipeline mirrors what Megatron's MoELayer fine-grained callables drive
through PrimusTurboDeepEPTokenDispatcher / PrimusTurboGroupedMLP:

    hidden ─► dispatch_preprocess ─► token_dispatch (DeepEP A2A) ─► dispatch_postprocess
           ─► grouped_gemm (fc1) ─► swiglu_with_probs ─► grouped_gemm (fc2)
           ─► combine_preprocess  ─► token_combine  (DeepEP A2A) ─► combine_postprocess
           ─► hidden

References:
    primus/backends/megatron/core/extensions/primus_turbo.py
        - PrimusTurboDeepEPTokenDispatcher (dispatch/combine split into 6 stages)
        - PrimusTurboGroupedMLP            (fc1 + swiglu_with_probs + fc2)
    primus/backends/megatron/patches/turbo/moe_dispatcher_patches.py
        - patches Megatron's MoEFlexTokenDispatcher to the Turbo version when
          enable_primus_turbo=True, use_turbo_deepep=True, TP=1.

Default model shape (from deepseek_v3.yaml + deepseek_v3-BF16-pretrain.yaml):
    hidden_size=7168, moe_ffn_hidden_size=2048, num_experts=256, topk=8,
    micro_batch_size=2, seq_length=4096, TP=1, EP=<world_size> (8 on a node).

Launch (single MI355X node, EP=8):

    torchrun --standalone --nproc-per-node=8 \
        benchmark/kernel/moe/bench_turbo_moe_e2e.py

Smaller smoke run on EP=2 (e.g. 2 GPUs):

    torchrun --standalone --nproc-per-node=2 \
        benchmark/kernel/moe/bench_turbo_moe_e2e.py \
        --num-experts 16 --topk 4 --seq-length 1024 --micro-batch-size 1
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

# All Turbo kernels live in primus_turbo.pytorch.{ops,modules}.
import primus_turbo.pytorch as turbo

# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark the Turbo MoE forward + backward pipeline "
            "(dispatch -> fc1 -> swiglu -> fc2 -> combine) against the "
            "deepseek_v3-BF16-pretrain.yaml shape."
        )
    )
    # Model shape (defaults match deepseek_v3 + MI355X pretrain yaml).
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--moe-ffn-hidden-size", type=int, default=2048)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Activation / weight dtype for the BF16 baseline.",
    )

    # DeepEP / dispatcher knobs (matches yaml).
    p.add_argument("--deepep-num-cu", type=int, default=80, help="turbo_deepep_num_cu")
    p.add_argument(
        "--deepep-use-comm-stream",
        action="store_true",
        help="turbo_deepep_use_comm_stream (default off, like the yaml).",
    )
    p.add_argument(
        "--no-permute-fusion",
        action="store_true",
        help="Disable moe_permute_fusion (default on, like the yaml).",
    )
    p.add_argument(
        "--no-cuda-tokens-per-expert",
        action="store_true",
        help=(
            "Return tokens_per_expert on CPU. Default = CUDA tensor "
            "(matches use_turbo_grouped_mlp=True path; required for "
            "sync-free / num_worst_tokens > 0)."
        ),
    )
    p.add_argument(
        "--sync-free-stage",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=(
            "turbo_sync_free_moe_stage. >1 enables deepep_num_worst_tokens; "
            ">2 enables permute_max_token_num (fully sync-free)."
        ),
    )

    # Benchmark loop.
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--mode",
        type=str,
        default="fwd",
        choices=["fwd", "fwd_bwd"],
        help="Whether to also time the backward pass. (sweep tables only use fwd.)",
    )
    p.add_argument(
        "--no-force-balance",
        action="store_true",
        help=(
            "Disable load-balancing. Probs are uniform 1/topk so torch.topk "
            "picks experts 0..topk-1 for every token (very unrealistic, but "
            "available for completeness)."
        ),
    )
    p.add_argument(
        "--routing",
        type=str,
        default="spread",
        choices=["spread", "cluster", "random"],
        help=(
            "Force-balanced routing pattern. "
            "'cluster' = the pattern PrimusTurboDeepEPTokenDispatcher uses "
            "when moe_router_force_load_balancing=True: "
            "token i -> experts [i*topk, i*topk+1, ..., i*topk+topk-1] mod E. "
            "Because num_local_experts == E/EP and the topk experts are "
            "contiguous, every token is dispatched to only 1 remote rank, "
            "which underestimates dispatch/combine A2A volume by ~ep_size. "
            "'spread' (default) = token i -> experts (i + j*num_local_experts) "
            "mod E, so the topk experts span all EP ranks (1 per rank). "
            "This matches the expected per-rank A2A volume under realistic "
            "balanced routing. "
            "'random' = sample topk distinct experts per token uniformly."
        ),
    )

    # Sweep mode: produces a per-batch-size breakdown table like the
    # reference DeepSeek-V3 dispatch table the user shared.
    p.add_argument(
        "--sweep",
        type=str,
        default=None,
        help=(
            "Comma-separated list of num_tokens-per-rank values to sweep "
            "(e.g. '1,2,4,8,16,32,64,128,4096,8192'). Each value sets "
            "mbs=1, seq_length=value so num_tokens=value. Overrides "
            "--micro-batch-size and --seq-length."
        ),
    )

    # Output.
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV file path for rank-0 results.",
    )
    p.add_argument("--seed", type=int, default=2026)

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dtype_from_str(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


@dataclass
class StageTimer:
    """Aggregates CUDA-event timings for one named stage."""

    name: str
    start_events: List[torch.cuda.Event] = field(default_factory=list)
    end_events: List[torch.cuda.Event] = field(default_factory=list)

    def begin(self):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.start_events.append(ev)

    def end(self):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.end_events.append(ev)

    def samples_ms(self) -> List[float]:
        return [s.elapsed_time(e) for s, e in zip(self.start_events, self.end_events)]

    def reset(self):
        self.start_events.clear()
        self.end_events.clear()


def _all_reduce_scalar(value: float, op=dist.ReduceOp.AVG) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return t.item()


# ---------------------------------------------------------------------------
# Turbo MoE module
# ---------------------------------------------------------------------------


class TurboMoELayer(torch.nn.Module):
    """Single MoE layer that wires the Turbo kernels exactly like the patched
    Megatron path (dispatch -> fc1 -> swiglu -> fc2 -> combine).

    Forward returns the final hidden_states with shape (num_tokens, hidden).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        moe_ffn_hidden_size: int,
        num_experts: int,
        topk: int,
        dtype: torch.dtype,
        ep_group: dist.ProcessGroup,
        num_tokens: int,
        deepep_num_cu: int,
        deepep_use_comm_stream: bool,
        permute_fusion: bool,
        deepep_use_cuda_num_tokens_per_expert: bool,
        sync_free_stage: int,
        force_balance: bool,
        routing: str = "spread",
    ):
        super().__init__()
        ep_size = ep_group.size()
        assert num_experts % ep_size == 0, (
            f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
        )

        self.hidden_size = hidden_size
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.dtype = dtype
        self.num_tokens = num_tokens
        self.force_balance = force_balance
        self.routing = routing
        self.permute_fusion = permute_fusion
        self.num_local_experts = num_experts // ep_size
        self.ep_size = ep_size

        # ---- sync-free MoE knobs (matches PrimusTurboDeepEPTokenDispatcher) ----
        num_worst_tokens = 0
        permute_max_token_num = 0
        if sync_free_stage > 1:
            # Mirror primus_turbo.py:1456-1466. tp_ep_size == ep_size here (tp=1).
            num_worst_tokens = num_tokens * ep_size
            if sync_free_stage > 2:
                permute_max_token_num = num_worst_tokens * topk

        # ---- dispatcher ----
        self.dispatcher = turbo.modules.DeepEPTokenDispatcher(
            num_experts=num_experts,
            router_topk=topk,
            ep_group=ep_group,
            tp_group=None,           # TP=1
            tp_ep_group=None,        # falls back to ep_group internally
            expert_capacity_factor=None,
            permute_fusion=permute_fusion,
            permute_max_token_num=permute_max_token_num,
            deepep_async_finish=True,
            deepep_allocate_on_comm_stream=True,
            deepep_use_comm_stream=deepep_use_comm_stream,
            deepep_num_use_cu=deepep_num_cu,
            deepep_num_worst_tokens=num_worst_tokens,
            deepep_use_cuda_num_tokens_per_expert=deepep_use_cuda_num_tokens_per_expert,
        )

        # ---- expert weights ----
        # PrimusTurboGroupedMLP._stack_grouped_linear_weight stacks each
        # per-expert linear weight along dim 0 and transposes (1,2). The result
        # used by grouped_gemm (trans_b=False) is shape [G, K, N].
        #
        # fc1: K=hidden, N=2*ffn  (GLU doubles the intermediate dim)
        # fc2: K=ffn,    N=hidden
        gain = 1.0 / (hidden_size ** 0.5)
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_local_experts,
                hidden_size,
                2 * moe_ffn_hidden_size,
                dtype=dtype,
                device="cuda",
            ).normal_(0.0, gain)
        )
        gain2 = 1.0 / (moe_ffn_hidden_size ** 0.5)
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_local_experts,
                moe_ffn_hidden_size,
                hidden_size,
                dtype=dtype,
                device="cuda",
            ).normal_(0.0, gain2)
        )

    # ---- six dispatcher/combine stages, individually exposed for per-stage timing ----

    def _build_token_indices(self, num_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        """Build token_indices for the DeepEP dispatcher.

        The choice of routing pattern strongly affects the dispatch / combine
        A2A volume, so we expose three patterns:

        * `cluster` reproduces what
          PrimusTurboDeepEPTokenDispatcher.dispatch_preprocess does when
          `moe_router_force_load_balancing=True`:
              token i -> experts [i*topk + 0, ..., i*topk + topk-1] mod E
          Because the topk experts are contiguous and num_local_experts = E/EP,
          every token is dispatched to only ONE remote rank. This is "balanced
          per expert" but artificially shrinks the inter-rank A2A volume by
          ~ep_size, which makes dispatch / combine look much faster than
          realistic routing in production.

        * `spread` (default) keeps the per-expert balance but rotates the
          choices so that each token's topk experts land on topk distinct
          ranks (one per rank when topk == ep_size):
              token i, slot j -> expert (i + j*num_local_experts) mod E
          This matches the inter-rank A2A volume one would see under a real
          balanced router (probs ~ uniform after the aux-loss/no-aux trick).

        * `random` uniformly samples topk distinct experts per token. Good for
          measuring noisy / imbalanced behavior, but groups will be uneven.
        """
        if not self.force_balance:
            return None

        if self.routing == "cluster":
            return (
                torch.arange(num_tokens * self.topk, device=device).view(num_tokens, self.topk)
                % self.num_experts
            )
        if self.routing == "spread":
            # token i, slot j -> (i + j*num_local_experts) mod num_experts
            i = torch.arange(num_tokens, device=device).view(num_tokens, 1)
            j = torch.arange(self.topk, device=device).view(1, self.topk)
            return (i + j * self.num_local_experts) % self.num_experts
        if self.routing == "random":
            # Uniform random topk distinct experts per token.
            # rand-then-topk is a standard Gumbel-style trick.
            scores = torch.rand((num_tokens, self.num_experts), device=device)
            return torch.topk(scores, self.topk, dim=-1).indices
        raise ValueError(f"Unknown routing pattern: {self.routing}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        stage_timer: Optional[dict] = None,
    ) -> torch.Tensor:
        assert hidden_states.shape == (self.num_tokens, self.hidden_size)
        assert probs.shape == (self.num_tokens, self.num_experts)

        token_indices = self._build_token_indices(self.num_tokens, hidden_states.device)

        def _maybe(name: str, fn: Callable):
            if stage_timer is None:
                return fn()
            t: StageTimer = stage_timer[name]
            t.begin()
            out = fn()
            t.end()
            return out

        # ---- DISPATCH ----------------------------------------------------------
        hidden_states, probs_topk = _maybe(
            "dispatch_preprocess",
            lambda: self.dispatcher._pre_dispatch(
                hidden_states, probs, routing_map=None, token_indices=token_indices
            ),
        )
        dispatched_tokens, dispatched_probs = _maybe(
            "token_dispatch",
            lambda: self.dispatcher._exec_dispatch(hidden_states, probs_topk),
        )
        permuted_input, tokens_per_expert, permuted_probs = _maybe(
            "dispatch_postprocess",
            lambda: self.dispatcher._post_dispatch(dispatched_tokens, dispatched_probs),
        )

        # tokens_per_expert is int64; grouped_gemm wants int64 group_lens.
        # When CUDA tokens_per_expert is enabled it's already on device.
        if tokens_per_expert.device.type != "cuda":
            tokens_per_expert = tokens_per_expert.to(device="cuda", non_blocking=True)
        tokens_per_expert = tokens_per_expert.to(torch.int64)

        # ---- FC1 ---------------------------------------------------------------
        fc1_output = _maybe(
            "grouped_gemm_fc1",
            lambda: turbo.ops.grouped_gemm(
                permuted_input,
                self.w1,
                tokens_per_expert,
                trans_b=False,
            ),
        )

        # ---- SwiGLU + probs ----------------------------------------------------
        # Mirrors PrimusTurboGroupedMLP._activation_func_with_probs (primus_turbo.py:1299-1306):
        #   row_mask = tokens_per_expert_to_mask(tokens_per_expert, num_tokens)
        #   out      = swiglu_with_probs(fc1_output, permuted_probs, row_mask)
        def _act():
            row_mask = turbo.ops.tokens_per_expert_to_mask(
                tokens_per_expert, fc1_output.shape[0]
            )
            return turbo.ops.swiglu_with_probs(fc1_output, permuted_probs, row_mask)

        activated = _maybe("swiglu_with_probs", _act)

        # ---- FC2 ---------------------------------------------------------------
        fc2_output = _maybe(
            "grouped_gemm_fc2",
            lambda: turbo.ops.grouped_gemm(
                activated,
                self.w2,
                tokens_per_expert,
                trans_b=False,
            ),
        )

        # ---- COMBINE -----------------------------------------------------------
        combine_input = _maybe(
            "combine_preprocess",
            lambda: self.dispatcher._pre_combine(fc2_output),
        )
        combined = _maybe(
            "token_combine",
            lambda: self.dispatcher._exec_combine(combine_input),
        )
        output = _maybe(
            "combine_postprocess",
            lambda: self.dispatcher._post_combine(combined),
        )

        return output


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


STAGE_NAMES = [
    "dispatch_preprocess",
    "token_dispatch",
    "dispatch_postprocess",
    "grouped_gemm_fc1",
    "swiglu_with_probs",
    "grouped_gemm_fc2",
    "combine_preprocess",
    "token_combine",
    "combine_postprocess",
]

# Grouping used by the breakdown table that mirrors the DeepSeek-V3 dispatch
# reference table the user shared (columns: sort / dispatch / fused_moe /
# combine / misc / all_kernels).
#   sort      = router topk done inside _pre_dispatch
#   dispatch  = DeepEP A2A (moe_dispatch)
#   fused_moe = expert MLP (fc1 + swiglu + fc2)
#   combine   = DeepEP A2A (moe_combine)
#   misc      = permute / unpermute / indices_to_multihot / view
STAGE_GROUPS = {
    "sort": ["dispatch_preprocess"],
    "dispatch": ["token_dispatch"],
    "fused_moe": ["grouped_gemm_fc1", "swiglu_with_probs", "grouped_gemm_fc2"],
    "combine": ["token_combine"],
    "misc": ["dispatch_postprocess", "combine_preprocess", "combine_postprocess"],
}


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    hidden_states.requires_grad_(requires_grad)

    # Uniform probs so the topk weights sum to 1.0 per token. This matches the
    # synthetic forced-balance setup used in primus_turbo's own dispatcher tests
    # (tests/pytorch/modules/test_token_dispatcher.py).
    probs = torch.full(
        (num_tokens, num_experts),
        1.0 / topk,
        dtype=torch.float32,
        device=device,
    )
    return hidden_states, probs


def _format_stage_table(per_stage_ms: dict, mode: str) -> str:
    """ASCII table of per-stage timings."""
    header = ["Stage", "Mean (ms)", "Median (ms)", "Std (ms)", "Min (ms)", "Max (ms)"]
    rows = []
    total_mean = 0.0
    for name in STAGE_NAMES:
        samples = per_stage_ms.get(name, [])
        if not samples:
            continue
        mean_v = statistics.fmean(samples)
        med = statistics.median(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        total_mean += mean_v
        rows.append(
            [
                name,
                f"{mean_v:.3f}",
                f"{med:.3f}",
                f"{std:.3f}",
                f"{min(samples):.3f}",
                f"{max(samples):.3f}",
            ]
        )
    rows.append(["[sum-of-stages]", f"{total_mean:.3f}", "-", "-", "-", "-"])

    col_widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    sep = "+".join("-" * (w + 2) for w in col_widths)

    def _fmt(row):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |"

    lines = [f"\n[{mode}] per-stage timing (CUDA events, this rank):"]
    lines.append(sep)
    lines.append(_fmt(header))
    lines.append(sep)
    for r in rows:
        lines.append(_fmt(r))
    lines.append(sep)
    return "\n".join(lines)


def _dump_csv(path: str, rank: int, world_size: int, args, e2e_ms: dict, per_stage_ms: dict):
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds")
    fieldnames = [
        "timestamp",
        "rank",
        "world_size",
        "hidden_size",
        "moe_ffn_hidden_size",
        "num_experts",
        "topk",
        "micro_batch_size",
        "seq_length",
        "num_tokens",
        "dtype",
        "deepep_num_cu",
        "permute_fusion",
        "deepep_use_cuda_num_tokens_per_expert",
        "sync_free_stage",
        "mode",
        "stage",
        "mean_ms",
        "median_ms",
        "p99_ms",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        base = {
            "timestamp": ts,
            "rank": rank,
            "world_size": world_size,
            "hidden_size": args.hidden_size,
            "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
            "num_experts": args.num_experts,
            "topk": args.topk,
            "micro_batch_size": args.micro_batch_size,
            "seq_length": args.seq_length,
            "num_tokens": args.micro_batch_size * args.seq_length,
            "dtype": args.dtype,
            "deepep_num_cu": args.deepep_num_cu,
            "permute_fusion": not args.no_permute_fusion,
            "deepep_use_cuda_num_tokens_per_expert": not args.no_cuda_tokens_per_expert,
            "sync_free_stage": args.sync_free_stage,
            "mode": args.mode,
        }

        for stage_name, samples in {**e2e_ms, **per_stage_ms}.items():
            if not samples:
                continue
            mean_v = statistics.fmean(samples)
            med = statistics.median(samples)
            sorted_s = sorted(samples)
            p99 = sorted_s[max(0, int(0.99 * (len(sorted_s) - 1)))]
            writer.writerow(
                {**base, "stage": stage_name, "mean_ms": mean_v, "median_ms": med, "p99_ms": p99}
            )


@dataclass
class RunResult:
    """Aggregated mean (ms) per stage for one (num_tokens) point."""

    num_tokens: int
    per_stage_mean_ms: dict
    e2e_forward_mean_ms: float
    e2e_backward_mean_ms: Optional[float]
    raw_per_stage_ms: dict
    raw_e2e_ms: dict


def _run_one_point(
    *,
    args,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
    ep_group,
    num_tokens: int,
) -> RunResult:
    """Build the Turbo MoE module for one num_tokens point and time it."""

    moe = TurboMoELayer(
        hidden_size=args.hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        num_experts=args.num_experts,
        topk=args.topk,
        dtype=dtype,
        ep_group=ep_group,
        num_tokens=num_tokens,
        deepep_num_cu=args.deepep_num_cu,
        deepep_use_comm_stream=args.deepep_use_comm_stream,
        permute_fusion=not args.no_permute_fusion,
        deepep_use_cuda_num_tokens_per_expert=not args.no_cuda_tokens_per_expert,
        sync_free_stage=args.sync_free_stage,
        force_balance=not args.no_force_balance,
        routing=args.routing,
    ).cuda()

    do_bwd = args.mode == "fwd_bwd"
    hidden_states, probs = _make_inputs(
        num_tokens=num_tokens,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        topk=args.topk,
        dtype=dtype,
        device=device,
        requires_grad=do_bwd,
    )

    # warmup
    for _ in range(args.warmup):
        out = moe(hidden_states, probs)
        if do_bwd:
            out.sum().backward()
            moe.w1.grad = None
            moe.w2.grad = None
            if hidden_states.grad is not None:
                hidden_states.grad = None
    torch.cuda.synchronize()
    dist.barrier()

    stage_timer = {name: StageTimer(name) for name in STAGE_NAMES}
    e2e_fwd = StageTimer("e2e_forward")
    e2e_bwd = StageTimer("e2e_backward") if do_bwd else None

    for _ in range(args.iters):
        e2e_fwd.begin()
        out = moe(hidden_states, probs, stage_timer=stage_timer)
        e2e_fwd.end()

        if do_bwd:
            grad_out = torch.randn_like(out)
            e2e_bwd.begin()
            out.backward(grad_out)
            e2e_bwd.end()
            moe.w1.grad = None
            moe.w2.grad = None
            if hidden_states.grad is not None:
                hidden_states.grad = None

    torch.cuda.synchronize()

    per_stage_ms = {name: stage_timer[name].samples_ms() for name in STAGE_NAMES}
    e2e_ms = {"e2e_forward": e2e_fwd.samples_ms()}
    if do_bwd:
        e2e_ms["e2e_backward"] = e2e_bwd.samples_ms()

    per_stage_mean = {
        n: (statistics.fmean(per_stage_ms[n]) if per_stage_ms[n] else 0.0)
        for n in STAGE_NAMES
    }
    e2e_fwd_mean = statistics.fmean(e2e_ms["e2e_forward"]) if e2e_ms["e2e_forward"] else 0.0
    e2e_bwd_mean = (
        statistics.fmean(e2e_ms["e2e_backward"])
        if do_bwd and e2e_ms.get("e2e_backward")
        else None
    )

    # Free module / inputs / grads before the next sweep point.
    del moe, hidden_states, probs
    torch.cuda.empty_cache()
    dist.barrier()

    return RunResult(
        num_tokens=num_tokens,
        per_stage_mean_ms=per_stage_mean,
        e2e_forward_mean_ms=e2e_fwd_mean,
        e2e_backward_mean_ms=e2e_bwd_mean,
        raw_per_stage_ms=per_stage_ms,
        raw_e2e_ms=e2e_ms,
    )


def _compute_breakdown_row(args, dtype: torch.dtype, r: RunResult) -> dict:
    """Map per-stage means to the (Time / Compute / GB/s / sort / dispatch /
    fused_moe / combine / misc / all_kernels) columns from the reference table.
    """
    groups_us = {}
    for group, members in STAGE_GROUPS.items():
        groups_us[group] = sum(r.per_stage_mean_ms[m] * 1000.0 for m in members)

    all_kernels_us = sum(groups_us.values())
    time_us = r.e2e_forward_mean_ms * 1000.0

    # Compute (FC1 + FC2 fwd only): 6 * dispatched_tokens * hidden * ffn FLOPs.
    dispatched_tokens = r.num_tokens * args.topk  # per rank under force-balance
    fc_flops_fwd = 6.0 * dispatched_tokens * args.hidden_size * args.moe_ffn_hidden_size
    compute_tflops = (fc_flops_fwd / (time_us * 1e-6) / 1e12) if time_us > 0 else 0.0

    # Global memory traffic estimate (bytes / iter).
    #   FC1+FC2 weights (loaded each iter, per local expert):
    bytes_per = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    num_local_experts = args.num_experts // dist.get_world_size()
    weight_bytes = (
        num_local_experts
        * (
            args.hidden_size * 2 * args.moe_ffn_hidden_size
            + args.moe_ffn_hidden_size * args.hidden_size
        )
        * bytes_per
    )
    #   FC1+FC2 activations (read once, write once per stage):
    fc_act_bytes = (
        dispatched_tokens
        * (args.hidden_size + 2 * args.moe_ffn_hidden_size + args.moe_ffn_hidden_size + args.hidden_size)
        * bytes_per
    )
    #   DeepEP dispatch + combine inter-rank payload (rough):
    deepep_bytes = 2 * r.num_tokens * args.topk * args.hidden_size * bytes_per
    total_bytes = weight_bytes + fc_act_bytes + deepep_bytes
    gbps = (total_bytes / (time_us * 1e-6) / 1e9) if time_us > 0 else 0.0

    return {
        "num_tokens": r.num_tokens,
        "time_us": time_us,
        "tflops": compute_tflops,
        "gbps": gbps,
        "sort_us": groups_us["sort"],
        "dispatch_us": groups_us["dispatch"],
        "fused_moe_us": groups_us["fused_moe"],
        "combine_us": groups_us["combine"],
        "misc_us": groups_us["misc"],
        "all_kernels_us": all_kernels_us,
    }


def _format_breakdown_table(rows: List[dict], dtype_str: str) -> str:
    header = [
        "Batch Size",
        "Time (us)",
        "Compute (TFLOPS)",
        "Global Memory (GB/s)",
        "sort (us)",
        "dispatch (us)",
        "fused_moe (us)",
        "combine (us)",
        "misc (us)",
        "all_kernels (us)",
    ]
    import math as _math

    def _fmt_num(value, spec):
        if value is None or (isinstance(value, float) and _math.isnan(value)):
            return "-"
        return format(value, spec)

    body = []
    for row in rows:
        body.append(
            [
                f"{row['num_tokens']}",
                _fmt_num(row["time_us"], ".1f"),
                _fmt_num(row["tflops"], ".1f"),
                _fmt_num(row["gbps"], ".0f"),
                _fmt_num(row["sort_us"], ".1f"),
                _fmt_num(row["dispatch_us"], ".1f"),
                _fmt_num(row["fused_moe_us"], ".1f"),
                _fmt_num(row["combine_us"], ".1f"),
                _fmt_num(row["misc_us"], ".1f"),
                _fmt_num(row["all_kernels_us"], ".1f"),
            ]
        )

    col_widths = [max(len(r[i]) for r in body + [header]) for i in range(len(header))]
    sep = "+".join("-" * (w + 2) for w in col_widths)

    def _fmt(row, align="right"):
        cells = []
        for c, w in zip(row, col_widths):
            cells.append(c.rjust(w) if align == "right" else c.ljust(w))
        return "| " + " | ".join(cells) + " |"

    lines = [f"\nTurbo MoE forward breakdown (dispatch={dtype_str}, per rank):"]
    lines.append(sep)
    lines.append(_fmt(header, align="left"))
    lines.append(sep)
    for r in body:
        lines.append(_fmt(r))
    lines.append(sep)
    return "\n".join(lines)


def _dump_breakdown_csv(path: str, args, world_size: int, rows: List[dict]):
    fieldnames = [
        "timestamp",
        "world_size",
        "hidden_size",
        "moe_ffn_hidden_size",
        "num_experts",
        "topk",
        "dtype",
        "num_tokens",
        "time_us",
        "tflops",
        "gbps",
        "sort_us",
        "dispatch_us",
        "fused_moe_us",
        "combine_us",
        "misc_us",
        "all_kernels_us",
    ]
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": ts,
                    "world_size": world_size,
                    "hidden_size": args.hidden_size,
                    "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
                    "num_experts": args.num_experts,
                    "topk": args.topk,
                    "dtype": args.dtype,
                    **row,
                }
            )


def main():  # noqa: C901 (benchmark glue)
    args = _build_parser().parse_args()
    dtype = _dtype_from_str(args.dtype)

    # ------------------- torch.distributed ----------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if "RANK" not in os.environ:
        raise RuntimeError(
            "This benchmark must be launched with torchrun "
            "(it needs EP world_size > 1 for DeepEP A2A)."
        )
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep_group = dist.group.WORLD

    torch.manual_seed(args.seed + rank)

    sweep_points: List[int]
    if args.sweep is not None:
        sweep_points = [int(x) for x in args.sweep.split(",") if x.strip()]
    else:
        sweep_points = [args.micro_batch_size * args.seq_length]

    if rank == 0:
        print(
            f"[bench_turbo_moe_e2e] world_size={world_size} EP={world_size} TP=1 dtype={args.dtype} "
            f"hidden={args.hidden_size} ffn={args.moe_ffn_hidden_size} "
            f"E={args.num_experts} topk={args.topk} "
            f"sync_free_stage={args.sync_free_stage} permute_fusion={not args.no_permute_fusion} "
            f"deepep_num_cu={args.deepep_num_cu} routing={args.routing}"
        )
        print(f"[bench_turbo_moe_e2e] sweep num_tokens points: {sweep_points}")

    # ------------------- run sweep ------------------------------------------
    breakdown_rows = []
    for num_tokens in sweep_points:
        if rank == 0:
            print(f"\n[bench_turbo_moe_e2e] ==> num_tokens={num_tokens}")
        try:
            res = _run_one_point(
                args=args,
                dtype=dtype,
                device=device,
                rank=rank,
                world_size=world_size,
                ep_group=ep_group,
                num_tokens=num_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            # Some sweep points (very small BS) hit CK / DeepEP edge cases
            # (invalid kernel launch, empty groups, etc). Record the failure
            # in the breakdown table instead of aborting the whole sweep.
            if rank == 0:
                import traceback as _tb

                print(
                    f"[bench_turbo_moe_e2e] num_tokens={num_tokens} FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )
                _tb.print_exc()
            breakdown_rows.append(
                {
                    "num_tokens": num_tokens,
                    "time_us": float("nan"),
                    "tflops": float("nan"),
                    "gbps": float("nan"),
                    "sort_us": float("nan"),
                    "dispatch_us": float("nan"),
                    "fused_moe_us": float("nan"),
                    "combine_us": float("nan"),
                    "misc_us": float("nan"),
                    "all_kernels_us": float("nan"),
                }
            )
            # Try to keep the process group healthy for the next point.
            torch.cuda.synchronize()
            try:
                dist.barrier()
            except Exception:
                pass
            continue

        # Print per-rank per-stage table only when not sweeping (too noisy).
        if rank == 0 and args.sweep is None:
            print(_format_stage_table(
                {n: res.raw_per_stage_ms[n] for n in STAGE_NAMES},
                args.mode,
            ))
            print(f"[e2e] forward  mean (ms, rank0): {res.e2e_forward_mean_ms:.3f}")
            if res.e2e_backward_mean_ms is not None:
                print(f"[e2e] backward mean (ms, rank0): {res.e2e_backward_mean_ms:.3f}")

        row = _compute_breakdown_row(args, dtype, res)
        breakdown_rows.append(row)

        if rank == 0 and args.output_csv:
            _dump_csv(args.output_csv, rank, world_size, args, res.raw_e2e_ms, res.raw_per_stage_ms)

    # ------------------- print breakdown table (rank 0) ---------------------
    if rank == 0:
        print(_format_breakdown_table(breakdown_rows, args.dtype))
        if args.output_csv:
            csv_path = args.output_csv.replace(".csv", "_breakdown.csv") if args.output_csv.endswith(".csv") else args.output_csv + ".breakdown.csv"
            _dump_breakdown_csv(csv_path, args, world_size, breakdown_rows)
            print(f"\n[csv] wrote breakdown to {csv_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
