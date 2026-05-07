# 为什么 Primus 每 step 多花 70 ms 做 attention_mask, NeMo 却几乎免费

时间: 2026-04-30 18:05 CST

## 0. 起点疑问

Primus 和 NeMo 的 SFT dataset 代码**逐字一致**:

```python
@torch.no_grad()
def _create_attention_mask(self, max_length):
    attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
    attention_mask = attention_mask < 0.5
    return attention_mask

# collate_fn:
if not self.get_attention_mask_from_fusion:
    attention_mask = [self._create_attention_mask(max_length) for _ in batch]
    attention_mask = torch.stack(attention_mask)
```

但是:
- **Primus** trace 里 `aten::tril` 29 ms + `aten::ones` 14 ms + `aten::lt` 21 ms + `aten::stack` 6 ms ≈ **70 ms / step**
- **NeMo** trace 里 `aten::tril` × **0**, `aten::ones` × **0**, `aten::stack` × **0**

两边 attention_mask 在 forward 里同样被 `skip_getting_attention_mask_from_dataset=True` 直接置 None, 所以**这块 CPU 工作两边都是浪费**。区别只在 **谁付了浪费的成本**。

---

## 1. 关键差异: dataloader worker 数量

| 配置 | NeMo | Primus |
|---|---|---|
| `num_workers` | **8** (`NUM_WORKERS` env, default 8) | **0** (硬编码, 见 `llama2_custom.py:557`) |
| `persistent_workers` | True | False |
| collate 跑在哪 | 8 个 fork 出来的 worker 子进程 | trainer 主进程 |
| 主进程 trace 看得到 collate ops 吗? | **看不到** (worker 不在 kineto profiler 范围) | **能看到, 卡 70 ms** |

**Kineto profiler 只 instrument 主进程**, worker 子进程里的 `tril/ones/stack` 完全不出现在 trace 里, 但实际 CPU 工作是真实做了的; 只是被 prefetch 的 8 路并行掩盖, 不在主线程 critical path 上。

---

## 2. 为什么 Primus 不能也用 num_workers > 0?

之前实测 (`llama2_custom.py:548-557`):
- `num_workers=4, persistent=True` → SIGKILL on rank 5 after 2 min
- `num_workers=2, persistent=False` → SIGKILL on rank 7 after 2 min
- `num_workers=0` → 1024 steps OK

dmesg 显示是 **ROCm SVM/HMM (heterogeneous memory mgr) 的 page tables 把 host RAM 挤爆**:
- 8 个 main rank 各 ~131 GB anon-rss → 主进程合计 ~1.05 TB
- 内核 inactive_anon ~389 GB / NUMA node × 8 ≈ 3.1 TB
- 物理 RAM ~3 TB, swap=0 → fork() 触发 COW 时复制 SVM range, 撑破上限

NeMo 同机器同 GPU 用 `num_workers=8` **不会 OOM**, 说明问题不在内核而在主进程占用。

---

## 3. 为什么 Primus 主进程占用更高?

实测 VRAM (相同 LLaMA-2 70B + LoRA + BF16/FP8):

| | rank 0 max-allocated | reserved |
|---|---:|---:|
| NeMo (24h ago) | **214.65 GB** | 221.08 GB |
| Primus (今天 maskcache) | **285.84 GB** | 295.52 GB |
| Δ | **+71 GB (+33%)** | +74 GB |

VRAM 大 ⇒ ROCm 在 host 上为 GPU heap 维护的 SVM shadow / page tables 更大 ⇒ anon-rss 同步膨胀 ⇒ fork worker 时 COW 抖动更严重。

Primus 多占 71 GB 的可能源头 (待确认):
1. `use_distributed_optimizer=True` + `grad_reduce_in_fp32=True` + `fp8_param_gather=True` 的组合相比 NeMo 的配置, 有更多 copy buffer
2. Megatron-Bridge 的 `param_and_grad_buffer` 实现与 NeMo Lightning 的 `MegatronStrategy` 不同, bucket 策略不同
3. CUDA graph (Primus 启用) 会保留更多额外显存

---

## 4. 三条潜在解决方案

### A. 真正干掉这 70 ms (推荐)

既然 dataset 端 mask **forward 根本不消费**, 直接让 dataset 不生成:

```python
# llama2_custom.py 里, 给 FinetuningDatasetConfig 传:
dataset_kwargs={
    "return_cu_seqlen": False,
    "get_attention_mask_from_fusion": True,   # 让 collate_fn 走 else: attention_mask = None
}
```

这条路不需要 worker, 直接零成本; 已通过 `skip_getting_attention_mask_from_dataset=True` 验证 forward 不依赖该 tensor。

### B. 目前的 maskcache (subset of A)

只 cache mask 的 `tril+ones+lt` 结果, `collate_fn` 仍然 stack。
实测 -156 ms/step (-9.7%), TFLOPS 2272 → 2517 (+10.8%)。
是 A 的中间版本, 数学上等价 (`stack` 后是新的 contiguous tensor, 没有别名风险)。

### C. 修 Primus 显存让 num_workers > 0 可用

需要先定位 71 GB 的来源 (TE buffer? CUDA graph mempool? distributed-optimizer workspace?)。
风险高、收益不确定, 不优先做。

---

## 5. 结论

**"NeMo mask 开销小" 是观察错误 — NeMo 同样做了 70 ms 的浪费工作, 只是把它藏在 8 个 dataloader worker 子进程里, 主进程 trace 看不到。**

Primus 之所以暴露在主线程 critical path 上, 是因为:
1. 主进程 VRAM 比 NeMo 高 33% (286 vs 215 GB)
2. → fork dataloader worker 时 ROCm SVM page table 复制把 host RAM 撑爆
3. → 被迫退到 `num_workers=0`, collate 必须在主线程同步跑

正确解法不是 cache, 而是直接让 dataset **不生成这个 forward 不消费的 tensor**:
传 `dataset_kwargs={"get_attention_mask_from_fusion": True}` 到 `FinetuningDatasetConfig`,
让 `collate_fn` 走 `else: attention_mask = None` 分支。
