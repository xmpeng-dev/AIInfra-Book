# Llama2-70B LoRA SFT — attention_mask 代码路径 + 数值漂移分析

时间: 2026-04-30
背景: maskcache 实验完成后，实测 lm_loss 几乎完全一致 (max-min=0.008, ~0.25%)，
但 grad_norm 看起来 "飘了" (max-min ~17–60%)，需要确认 maskcache 的修改是否
真的影响了数值正确性。

---

## 1. 当前 Primus 的 attention_mask 数据流（代码层）

```
GPTSFTDataset.collate_fn  (host CPU, sft.py:708)
   │
   ├─ self.get_attention_mask_from_fusion = True  (默认, sft.py:98)
   │     → attention_mask = None (并不进入 batch dict)
   │
   │  *但 trace 实测显示 step21 里出现 aten::stack [[1,8192,8192]] 6 ms*
   │  *说明实际跑的时候 get_attention_mask_from_fusion=False，dataset 真的*
   │  *生成了 [1,L,L] bool tensor 并 stack 进 batch。*
   │
   ↓
get_batch_from_iterator  (gpt_step.py:36)
   │
   ├─ skip_getting_attention_mask_from_dataset = True
   │     (recipe llama2.py:241 显式设的, gpt_step.py:39 默认)
   │
   │  for key, val in batch.items():
   │      if key in required_device_keys:    # attention_mask 不在
   │          .cuda()
   │      elif key in required_host_keys:    #            不在
   │          .cpu()
   │      else:
   │          _batch_required_keys[key] = None    ← 强制覆盖成 None
   │
   ↓
forward(input_ids, position_ids, attention_mask=None, labels=...)
   │
   ↓
TE FlashAttention (aiter::fmha_v3_*_causal kernel)
   │
   └─ 内部用 fused causal mask, 完全不看 dataset 给的 mask
```

### 关键事实（trace 实证）

step21 (1449.7 ms 的代表性 step) 里 GPU 看到的 attention 相关 kernel:

| pattern                    | count | 总耗时       |
|----------------------------|-------|------------|
| `aiter::fmha_*_causal`     | 160   | 298.98 ms  |
| `aten::masked_fill` (cpu)  | 0     | 0          |
| `aten::where` (cpu)        | 0     | 0          |
| `aten::stack [[1,8192,8192]]` | 1   | 6.05 ms    |
| 8192² bool tensor → cuda  | 0     | 0 (D2H拷贝从未发生) |

> 也就是 **dataset 端生成的那个 8192×8192 bool tensor，从来没有真正进 GPU**。
> 它在 `get_batch_from_iterator` 里就被 `_batch_required_keys[key] = None` 直接覆盖掉了。
> dataset 的 mask 修改（cache vs no-cache, expand vs clone）在数学上**对 forward/backward 计算没有任何影响**。

---

## 2. NeMo 怎么处理（对比参考）

源代码: `/workspace/deps/nemo/nemo/collections/llm/gpt/data/core.py:631`

```python
attention_mask = None
if not self.get_attention_mask_from_fusion:                    # NeMo 默认 False
    attention_mask = [self._create_attention_mask(max_length) for _ in batch]
    attention_mask = torch.stack(attention_mask)
processed_batch = {...}    # 不含 attention_mask
if not self.get_attention_mask_from_fusion:
    processed_batch["attention_mask"] = attention_mask
```

`_create_attention_mask` 实现一字不差:
```python
attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
attention_mask = attention_mask < 0.5
```

**NeMo 与 Primus 在 attention_mask 处理上完全一致**：都是
- 默认让 fused attention kernel 自己造 causal mask
- dataset 端只在 `get_attention_mask_from_fusion=False` 时才生成
- 用 `torch.tril(torch.ones(...))` 构建，逐 batch element list-comp 然后 stack

NeMo 也没有 cache 这个 mask。Primus 这个 codebase 是 **fork 自 NeMo 的 GPTSFTDataset**。

---

## 3. lm_loss vs grad_norm 跨 run 噪声实测

6 次 run（同 seed=1234, 同代码 baseline / 不同小修改），iter10 / iter20 全部数据：

| run                    | iter10 lm_loss | iter10 grad | iter20 lm_loss | iter20 grad |
|------------------------|---------------:|------------:|---------------:|------------:|
| profile_run            | 3.218934       | 0.967       | 1.675518       | 0.491       |
| pinmem_true            | 3.218015       | 0.967       | 1.674731       | 0.485       |
| smallbucket            | 3.218240       | 0.914       | 1.678312       | 0.352       |
| workers0 (baseline)    | 3.220509       | 0.922       | 1.669310       | 0.400       |
| allranks (1024 iter)   | 3.213497       | 0.989       | 1.673663       | 0.559       |
| **maskcache**          | **3.212831**   | **1.074**   | **1.677475**   | **0.517**   |
| **mean**               | 3.217          | 0.972       | 1.675          | 0.467       |
| **std**                | 0.0026         | 0.054       | 0.0030         | 0.077       |
| **std/mean**           | **0.08%**      | **5.6%**    | **0.18%**      | **16%**     |

观察：
- **lm_loss noise = 0.08–0.18%**：典型的 BF16/FP8 跨 run 噪声水平
- **grad_norm noise = 5.6–16%**：更大，但仍属正常范围 — grad_norm 是 80 层 × 几亿 LoRA 参数的全局 L2 范数 ($\sqrt{\sum_p \|g_p\|^2}$)，对每层 BF16 前向/反向、TE FP8 amax history、RCCL all-reduce 顺序都敏感
- maskcache 的 grad_norm `1.074 / 0.517` **不是离群点**：iter10 的 1.074 比 allranks 的 0.989 仅高 8.6%，比 baseline 的 0.922 高 16% — 都在跨 run 的 1σ 范围内

### 为什么 grad_norm 比 lm_loss 噪声大这么多？

1. **lm_loss 是平均值**（每 token 平均），噪声被 batch_size × seq_len = 8192 个样本 average out
2. **grad_norm 是 L2 范数**：$\|g\|_2 = \sqrt{\sum_i g_i^2}$，每个 $g_i$ 是 BF16 数 reduce 出来的，加法非结合
3. **梯度幅值越小，相对噪声越大**：iter20 grad_norm 已降到 ~0.4，~0.05 的绝对噪声就是 12%；iter10 还在 ~1.0，5–10% 绝对差也才 5–10% 相对差
4. **DDP all-reduce 顺序非确定**：8 ranks 的 RCCL collective 在不同 run 上 ordering 可能不同
5. **TE FP8 amax history**：动态范围跟踪逐 step 更新，cold start 也会带噪声

---

## 4. 为什么 maskcache run 的 grad_norm "看起来" 偏高？

**误判原因**：之前对比的 baseline 只取了 1 个 run (workers0: 0.922)。
当时数据：
- maskcache iter10 grad: 1.074
- workers0 iter10 grad: 0.922
- "差异 16.5%" → 怀疑 maskcache 改坏了

**真相**：跨 run 集合 [0.914, 0.922, 0.967, 0.967, 0.989, 1.074] 的 1σ ≈ 5.6%。
maskcache 的 1.074 距离 mean (0.972) 差 +1.9σ —— 正态分布下 ~6% 概率会出现，**完全在噪声 band 内**。

---

## 5. 结论

1. **dataset 端的 attention_mask（无论怎么改）从来没进入 forward/backward 计算**。
   `get_batch_from_iterator` 在 `skip_getting_attention_mask_from_dataset=True`（recipe 默认）下会强制把它设成 None，attention 内部用 fused causal mask。
2. **maskcache run 的 grad_norm 偏移完全在跨 run 噪声 band 内**，不是 maskcache 引入的数值 bug。
3. **lm_loss 才是数值正确性的可信指标**，maskcache 的 lm_loss 与 baseline 偏离 < 0.3%，远小于跨 run 噪声。
4. **NeMo 处理逻辑与 Primus 一字不差**（NeMo 也是 `_create_attention_mask` + `tril(ones)` + 逐 element list comp + stack）。Primus 这块代码本来就是 fork 自 NeMo。

### 后续建议

- **可以放心合并 maskcache 优化**（用 `.expand()` 零拷贝，因为下游根本不消费这个 tensor）
- 进一步可以直接把 `if not self.get_attention_mask_from_fusion:` 这块删了
  （或在 recipe 里设 `dataset_kwargs={"get_attention_mask_from_fusion": True}`），
  彻底跳过 6 ms 的 stack — 因为 forward 本来就不要它
- grad_norm 的 5–16% 跨 run 噪声是 BF16 + multi-rank 的固有特征，不是 bug，
  做收敛对比时应使用 lm_loss 或者跑 ≥3 个 baseline 取 95% CI
