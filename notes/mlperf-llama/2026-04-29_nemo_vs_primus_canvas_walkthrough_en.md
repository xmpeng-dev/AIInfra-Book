# NeMo vs Primus - Llama-2-70B LoRA SFT (simple talk track)

> Short script for a meeting.  
> Only two points: **speed gap** and **memory gap**.

---

## 1) Speed gap: mostly DataLoader workers

I compared NeMo and Primus trace:

the summary is:

> Compute kernels are mostly aligned.  
> Right now, most of the performance gap is from the DataLoader pipeline.
> the memory gap is  from config


Main result:

> On the **same** hardware and the **same** workload, NeMo is **1490 ms** and Primus is **1626 ms** per step, so **NeMo is 8.4% faster**. NeMo's compute stream is 99.5% busy vs Primus' 84.9%

Why:
From the trace, the biggest reason is:

> Primus has a **191 ms idle gap** at step start.  
> This comes from single-process DataLoader work on CPU.  
> NeMo uses **8 persistent prefetch workers**, so this idle gap is gone.

What I did today:

- I tried to fix this in Primus by enabling DataLoader workers.  
- We hit a deadlock in the run, so we need more time to debug it.  
- This needs code changes in the Primus Bridge data path, not just one YAML line.

---

## 2) Memory gap: mostly config + workflow differences

The HBM gap is mostly from config differences.

There are many config differences, and I did not test them one by one.  
I'll share this file, and you can open it directly in Cursor.
- I mainly tried a few DDP-related options that NeMo does not use.  
- With those changes, Primus performance dropped by about 5%.  
- So we need more time to validate this, mainly around fp8 + ddp + te parameters, especially `fp8_param` (likely high impact, but not tested yet).

Model · fp8_param (likely high impact, but I have not had time to test it yet)

---

## 3) Detailed version (for a longer meeting)

> Use this part if you need a 5-7 minute update.  
> Same two topics: speed gap and memory gap, but with more detail.

### 3.1 Scope and fairness

I compared NeMo and Primus with the same base setup:

- same hardware: 8 x MI355X
- same workload: Llama-2-70B LoRA SFT
- same seq length: 8192 (packed)
- same data parallel world size: DP=8
- same main compute path (FP8 GEMM + CK V3 attention)

So this is not a model-quality difference.  
It is mostly a system/config/workflow difference.

### 3.2 Speed gap: where the 8.4% comes from

Main number:

> NeMo 1490 ms vs Primus 1626 ms, so NeMo is 8.4% faster per step.

From the trace, the biggest reason is:

> Primus has ~191 ms idle at the start of each step.

Why this happens:

- Primus baseline uses single-process DataLoader (`num_workers=0`)
- CPU does sync mask work on the hot path
- GPU waits for that CPU work before real compute starts

Why NeMo does not have this:

- NeMo uses 8 persistent prefetch workers
- next batch is prepared earlier
- step starts with compute, almost no idle bubble

What I already tried:

- I enabled DataLoader workers on Primus
- run hit deadlock (fork-after-CUDA type issue in this workflow)
- so this is not done yet

What is needed:

- code-level fix in Primus Bridge data workflow
- likely set and wire spawn context correctly
- then rerun and validate stability + speed

### 3.3 What is aligned at kernel level

Important note for discussion:

- the main compute kernels are already close/aligned
- the speed gap is not because GEMM is much weaker
- the biggest visible gap is the input pipeline idle

So the current result is:

> Core compute is mostly fine. Input pipeline is the bottleneck.

### 3.4 Memory gap: mostly config and workflow

HBM gap is mainly config-driven.  
But NeMo and Primus do not expose the same config surface.

That means:

- it is hard to do strict one-to-one config matching
- many items need code-path changes, not just YAML edits

Configs I called out first:

- `overlap_grad_reduce`
- `overlap_param_gather`
- `overlap_param_gather_with_optimizer_step`
- `average_in_collective`
- `gradient_reduce_div_fusion`
- `pad_buckets_for_high_nccl_busbw`

Current status on these:

- I tried a subset of DDP-style alignment changes
- with those changes, Primus speed dropped ~5%
- so this cannot be treated as “just make it same as NeMo”

For memory, the higher-impact area is likely fp8-related runtime config, especially:

- `fp8_param` (likely high impact for HBM)

But:

- I have not completed controlled A/B on this yet
- this needs more engineering time and clean reruns

### 3.5 Why this is taking time

NeMo and Primus use different training workflows:

- different runtime override points
- different config-to-runtime mapping
- some values are set inside code path, not visible in final YAML

So verification path is:

1. patch code path
2. rerun stable job
3. compare trace + step time + HBM
4. keep only changes that help both perf and stability

### 3.6 Next steps

Speed track:

- finish DataLoader worker fix in Primus workflow
- rerun and confirm the 191 ms idle shrink

Memory track:

- test high-impact fp8/ddp/te configs one by one
- include `fp8_param` A/B with clean measurement
- avoid forcing full parity if it hurts Primus speed

### 3.7 Detailed closing line (you can read this directly)

> In this comparison, NeMo is 8.4% faster per step, and most of that comes from Primus DataLoader idle time, not from core compute kernels.  
> I already tried to fix that path, but the run hit deadlock, so this needs more code-level work in the Primus workflow.  
> For memory, the gap is mainly config and workflow driven.  
> We should not force strict config parity, because that already showed a performance drop in Primus.  
> Next, I will finish the DataLoader fix and run focused A/B on high-impact fp8/ddp/te configs, especially `fp8_param`, then report a clean trade-off between speed and HBM.
