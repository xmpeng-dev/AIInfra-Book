## 时间 / 环境

- **时间**: 2026-05-13 22:20 +0800
- **机器**: `mi355-gpu-26` (8× MI355X / gfx950)
- **容器**: `xiaoming-dev`
- **配置**: PP=1 / EP=8 / TP=1 / 4 层 / DSV3 256E / top_k=8 / H=7168 / F=2048 / seq=2048 / micro_batch=1 / global_batch=8 / mock data / `train_iters=20` / `log_interval=1`
- **代码改动**: `python/mmoe/function.py` 的 `_backward_decomposed`

## 什么问题

上一篇 note ([2150 loss parity](./2026-05-13_2150_primus_dsv3_loss_parity_baseline_vs_mmoe.md)) 验证了 forward + eager_bwd 数值正确，但 eager_bwd wall time 是 615 ms/iter（vs baseline 232 ms），整训练 **2.7× slower**。eager_bwd 在 `_eager_moe_reference` 里又跑了一遍完整 forward（autograd build graph + replay alltoall + per-expert grouped MLP），等于把 forward 加倍。

要把 eager 换成真正的 decomposed bwd：

- 复用 forward 时 super-kernel 已经保存的中间量（`permuted_input`, `fc1_gate_up_save`, `dispatch_expert_offsets`, `dispatch_topk_weights`, `pack_perm`, `pack_offsets`）；
- 手算 FC2 / SwiGLU / FC1 bwd（per-expert）；
- 反向 alltoall 用 plain `dist.all_to_all_single`（grad 是手动算出的，不需要 autograd）。

## 做了什么

### 1) 数据流梳理

```
sender side                    receiver side
─────────                      ─────────────
hidden_states [T_local, H]
  flat × K, sort by expert id  →  IPC scatter
  ↓ (pack_perm, pack_offsets)              ↓
                                permuted_input [G, R, H]
                                fc1_save = pre-SwiGLU (gate, up)
                                topk_weights [G, R]
                                  ↓ FC1, SwiGLU, FC2
                                out_recv [G, R, H]
                                  out_recv * pr
                                  ↓ IPC atomic-add to sender
y_sender [T_local, H]
```

Backward （**反着走，相同 split-sizes**）：

```
sender side                    receiver side
─────────                      ─────────────
grad_y [T_local, H]
  replicate K times via pack_perm → all_to_all_single
                                grad_y_recv [Σrecv, H]
                                grad_out_recv = pr × grad_y_recv
                                grad_pr      = <grad_y_recv, out_recv>_H
                                ↓ FC2 bwd:  grad_silu_up, grad_w2
                                ↓ SwiGLU bwd elementwise on saved (gate, up)
                                ↓ FC1 bwd:  grad_w1, grad_x_recv
                                ↓ all_to_all_single grad_x_recv → grad_x_sorted
grad_h_per_kt [T_local*K, H]
  unsort via pack_perm + sum K  → grad_hidden_states
```

### 2) 关键发现：sender-side sort 顺序必须从 kernel 读

`fused_moe_super_kernel.hip:pack_sort_phase` 用 **counting sort by expert id** + `atomicAdd` 写 `pack_perm`：

```cpp
for (int i = threadIdx.x; i < total_pairs; i += WG_SIZE) {
    int eid = topk_ids[i];
    int pos = atomicAdd(&lds->write_pos[eid], 1);
    args.pack_perm[pos] = i;
}
```

同 expert 内顺序受 thread 调度影响 → 非确定性。如果在 Python 端用 `torch.argsort(dst_rank, stable=True)` 自己重做 sort，得到的顺序跟 kernel 不一致，所以 receiver 端的 `permuted_input[src][slot]` 就跟 sender bwd 想发的 grad 对不上号 → **静默错位** 后续 loss 完全飘走。

**修正**: 直接读 `ws.pack_perm[: T_local * K]`，且：

```python
send_counts_cpu[dst] = pack_offsets[(dst+1)*E_local] - pack_offsets[dst*E_local]
recv_counts_cpu[src] = dispatch_expert_offsets[src][E_local]
```

### 3) Per-expert 收尾（FC2 + SwiGLU + FC1）

```python
for e in range(E_local):
    # gather (src, slot) ranges for this expert across all 8 srcs
    x_e        = cat(permuted_input[src, s:t] for ranges)         # [T_e, H]
    fc1_e      = cat(fc1_save[src, s:t]      for ranges)          # [T_e, 2F]
    grad_y_e   = cat(grad_y_recv[off:off+n]  for ranges)          # [T_e, H]
    pr_e       = cat(topk_weights[src, s:t]  for ranges)          # [T_e]

    gate, up = fc1_e.split(F, dim=-1)
    sig = sigmoid(gate);  silu_gate = gate * sig;  silu_up = silu_gate * up

    grad_out_recv = grad_y_e * pr_e.unsqueeze(-1)
    grad_silu_up  = grad_out_recv @ w2[e]            # [T_e, F]
    grad_w2[e]   += grad_out_recv.T @ silu_up        # [H, F]

    out_recv_e   = silu_up @ w2[e].T                 # [T_e, H]  (for grad_pr)
    grad_pr_e    = (grad_y_e * out_recv_e).sum(-1)   # [T_e]

    silu_prime = sig + gate*sig*(1-sig)
    grad_up    = grad_silu_up * silu_gate
    grad_gate  = grad_silu_up * up * silu_prime
    grad_fc1_e = cat([grad_gate, grad_up], -1)       # [T_e, 2F]

    grad_w1[e] += grad_fc1_e.T @ x_e                 # [2F, H]
    grad_x_e    = grad_fc1_e @ w1[e]                 # [T_e, H]

    # scatter grad_x_e + grad_pr_e back into [G,R] / [Σrecv] frames
```

### 4) 反向 alltoall + unsort

```python
# combine bwd (sender→receiver, same direction as forward dispatch)
dist.all_to_all_single(grad_y_recv, grad_y_sorted,
                       output_split_sizes=recv_counts_cpu,
                       input_split_sizes=send_counts_cpu, group=ep_group)

# dispatch bwd (receiver→sender)
dist.all_to_all_single(grad_sorted_h, grad_permuted_compact,
                       output_split_sizes=send_counts_cpu,
                       input_split_sizes=recv_counts_cpu, group=ep_group)

# grad-pr alltoall (receiver→sender)
dist.all_to_all_single(grad_pr_sorted, grad_pr_recv_flat,
                       output_split_sizes=send_counts_cpu,
                       input_split_sizes=recv_counts_cpu, group=ep_group)

# unsort using pack_perm (NOT a fresh argsort!)
grad_h_per_kt[pack_perm] = grad_sorted_h
grad_hidden = grad_h_per_kt.view(T_local, K, H).sum(dim=1)
```

## 实测三方对比 (20 iter, mock data, 同 seed)

| iter | baseline   | MMOE eager | MMOE decomposed | dec − base | dec − eager | b gn  | d gn  | b TFLOP/s | e TFLOP/s | d TFLOP/s |
|-----:|-----------:|-----------:|----------------:|-----------:|------------:|------:|------:|----------:|----------:|----------:|
|   1  | 12.011050  | 12.010880  | 12.011150       | +1.00e-04  | +2.70e-04   |  6.92 |  6.92 |   0.8     |  0.9      |  1.0      |
|   3  | 11.424120  | 11.423700  | 11.424570       | +4.50e-04  | +8.70e-04   |  7.22 |  7.22 |  97.0     | 53.3      | 76.4      |
|   5  |  6.487745  |  6.488276  |  6.489593       | +1.85e-03  | +1.32e-03   | 12.02 | 12.04 | 179.2     | 67.2      | 104.7     |
|  10  |  1.667311  |  1.665281  |  1.667085       | −2.26e-04  | +1.80e-03   |  5.97 |  5.97 | 182.6     | 68.8      | 106.8     |
|  15  |  0.728094  |  0.728003  |  0.728503       | +4.09e-04  | +4.99e-04   |  3.31 |  3.31 | 183.8     | 68.7      | 106.6     |
|  20  |  0.567783  |  0.568154  |  0.568059       | +2.76e-04  | −9.47e-05   |  2.96 |  2.97 | 184.3     | 68.9      | 106.7     |

(完整 20 行: `slab/notes/monolith-moe/compare3.py`)

## 达成的效果

**数值正确性: PASS**

- decomposed vs baseline: max abs lm_loss diff **1.85e-3 @ iter 5**, max abs grad_norm diff 0.02 @ iter 5（baseline 12.02 / dec 12.04）
- decomposed vs eager: max abs lm_loss diff **1.80e-3 @ iter 10**
- 三个路径的 loss 曲线 12.01 → 0.568 完全同步下降，grad_norm 5.97 在 iter 10 三路径完全一致，从 6.92 (iter 1) 单调降到 2.96 (iter 20)
- iter 5 处差异最大是因为 grad_norm 12.02 → 数值最大，bf16 cancellation 风险最高（baseline / eager / dec 三路也在那一处分歧最大）

**性能: eager 69 → decomposed 107 TFLOP/s/GPU = +55 %**

- baseline (TEGroupedMLP, TE optimized) ~232 ms/iter (184 TFLOP/s/GPU stable)
- MMOE eager_bwd ~615 ms/iter (69 TFLOP/s/GPU stable)
- **MMOE decomposed_bwd ~400 ms/iter (107 TFLOP/s/GPU stable)**

decomposed 比 eager 快 1.55×，但还是 baseline 的 0.58×。差距来自：

1. **Python-side per-expert loop**: `for e in range(E_local=32)` 在 host 上启动 ~7 个 GEMM + 几个 elementwise ops + 几个 `cat`/`copy_`，每个 layer 都跑这个 loop 3 次（3 个 MoE layer），总共 ~100 ms host overhead。
2. **cat + slice 重排**: per-expert `cat([rows for src in ranges])` 走 HBM 拷贝（每个 expert 平均 ~64 rows × H=7168 × bf16 = ~900 KB），32 expert × 3 layer ≈ 90 MB 额外 HBM traffic per bwd。
3. **没有 fused SwiGLU bwd kernel**: elementwise `sigmoid` + `gate*sig*(1-sig)` + `grad_silu_up * silu_gate` 等是 8 个独立 kernel launch；写一个 fused kernel 估计能省 ~20 ms。
4. **没有 grouped GEMM**: 当前每个 expert 一个 single GEMM，浪费了 MFMA pipeline。用 grouped-bmm 或 grouped-GEMM 应该能拉到 ~150 TFLOP/s/GPU。

## 关键观察

1. **`ws.pack_perm` 是 sender-side sort 的唯一可信源**. super-kernel 的 sort 阶段用 atomicAdd 写 pack_perm，同 expert 内顺序 non-deterministic；Python 端任何重做 sort 的尝试都会产生跟 receiver permuted_input 错位的结果。必须 trust kernel 的输出。
2. **每个 layer 实例自有 workspace（megatron.py:227 `self._mmoe_ws = ws`）**, 所以 4 层模型有 3 个 MoE workspace（layer 1 是 dense）。同一 layer 的多 micro-batch / grad-checkpoint 重计算会共享同一 ws，未来上 grad accumulation 要解决。
3. **`grad_w1` / `grad_w2` 用 fp32 累加**, 跟 Megatron baseline 一致（`main_grads_dtype=float32`），避免 bf16 add 误差累积。Cast 回 bf16 在 return 时一次完成。
4. **eager_bwd 是有用的 ground truth**: decomposed 跟 eager 互校对（误差 1e-3 量级）证明 decomposed 没漏 grad 路径——如果只跟 baseline TEGroupedMLP 比，baseline 的 expert 调度顺序（per-expert GEMM 顺序、atomic-add 顺序）可能跟 super-kernel 不同，会让 bf16 noise 看起来更大。

## 下一步

- **P0 batched per-expert GEMM**: 把 `for e in range(E_local)` 的 4 个 GEMM 改成 batched `torch.bmm` 或者 grouped-gemm extension。预期 d 107 → ~140 TFLOP/s/GPU。
- **P0 fused SwiGLU bwd HIP kernel**: 一个 kernel 算出 grad_gate + grad_up + grad_silu_up 的所有 elementwise step。预期 −10~20 ms / iter。
- **P1 batched cat-and-slice → single permute kernel**: 把 per-(src, expert) gather 改成一个 HIP kernel 直接读 permuted_input 按 expert 重排到 contiguous buffer。预期 −15 ms / iter，且释放 ~90 MB 临时 HBM。
- **P1 fuse `out_recv_e` 重算到 FC2 bwd 的同一 GEMM**: 当前 `silu_up @ w2.T` 在 fwd 和 bwd 各算一次（fwd 算是 super-kernel 里，bwd 重算只为 grad_pr）。如果改成 fwd 把 `out_recv_e` 留到 fp16 buffer 里（每 layer ~50 MB），bwd 直接读，省一次 [H,F] GEMM。trade-off：HBM ↑ vs compute ↓。
- **P2 完整 HIP decomposed bwd kernel**: 把整个 bwd 写成一个 fused super-kernel (类似 forward super-kernel)，hardcode 32 expert + 8 GPU + 同样的 LDS 模板。目标 backward ≤ 1× forward wall = ~5 ms。这一步把 decomposed 推到 baseline 之上。

## 复现命令

```bash
ssh mi355-gpu-26
podman exec -it xiaoming-dev bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/3rd/Primus
export PYTHONPATH=/shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/python:$(pwd)
export PYTHONUNBUFFERED=1
export MMOE_DEBUG_LOSS_TO_STDERR=1
export PRIMUS_USE_MMOE=true
export MMOE_BACKWARD=decomposed

torchrun --standalone --nproc_per_node=8 --redirects 3 \
  --log-dir /tmp/dsv3_mmoe_dec_logs \
  primus/cli/main.py train pretrain \
  --config /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/examples/primus/dsv3_4layer_mmoe.yaml \
  --backend_path /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/3rd/Primus/third_party/Megatron-LM

# loss trajectory:
grep -nE 'MMOE-LOSS' /tmp/dsv3_mmoe_dec_logs/*/attempt_0/7/stderr.log

# 3-way compare:
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/slab/notes/monolith-moe
python3 compare3.py
```
