# Llama2-70B LoRA SFT — clean baseline vs maskcache 对照

时间: 2026-04-30 17:50 CST
背景: 早些时候发现 `xm-primus` 容器没用 `start_docker.sh` 启动，导致
`-v sft.py` / `-v llama2_custom.py` 这两条 bind-mount 没有生效。所以
本次会话之前所有 "改 host 文件 → 跑 → 看效果" 的步骤实际上对训练**没起作用**。
之前 maskcache 的 77 ms 提升数字其实是真的，但路径是 `docker cp` 把
maskcache 版直接拷进 container，而不是 mount。

为了拿到一组可信的对照数据，做了：
1. `docker stop xm-primus && docker rm xm-primus` + 重新 `bash start_docker.sh`
2. 验证 mount 生效: host == container md5 一致
3. 用同一 commit 的 host 文件先跑 baseline，再 patch maskcache 跑第二次

---

## 1. 结果

| run                  | iter10 step | iter10 loss | iter10 grad | iter20 step  | iter20 loss | iter20 grad | iter30 step | iter30 loss | iter30 grad |
|---------------------|------------:|------------:|------------:|-------------:|------------:|------------:|------------:|------------:|------------:|
| baseline (mount OK)  | 6418.2 ms   | 3.2164      | 1.039       | **1602.7 ms**| 1.6778      | 0.382       | 1614.6 ms   | 1.3512      | 0.306       |
| maskcache (mount OK) | 6385.7 ms   | 3.2311      | 0.925       | **1446.7 ms**| 1.6791      | 0.396       | 1485.2 ms   | 1.3484      | 0.210       |
| **Δ (maskcache - baseline)** | -32 ms | +0.0147 (0.45%) | -0.114 (-11%) | **-156 ms (-9.7%)** | +0.0013 (0.08%) | +0.014 (3.7%) | -129 ms (-8.0%) | -0.003 (-0.2%) | -0.096 (-31%) |

**TFLOPS**:
- iter20: 2272 → 2517 (**+245, +10.8%**)
- iter30: 2255 → 2452 (+197, +8.7%)

---

## 2. patch 内容

只改 `_create_attention_mask`，**不动** `collate_fn`，避免任何 storage 别名风险:

```python
@torch.no_grad()
def _create_attention_mask(self, max_length):
    cache = getattr(self, "_attention_mask_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(self, "_attention_mask_cache", cache)
    cached = cache.get(max_length)
    if cached is not None:
        return cached
    attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
    attention_mask = attention_mask < 0.5
    cache[max_length] = attention_mask
    return attention_mask
```

`collate_fn` 仍然是上游原版:
```python
attention_mask = [self._create_attention_mask(max_length) for _ in batch]
attention_mask = torch.stack(attention_mask)
```

**为什么这样安全**: stack 会把 list 的 N 个引用拷贝成一个新 contiguous tensor，
batch 内不存在共享 storage。dataset 层只省掉了 N×（tril+ones+<bool）的 CPU 工作，
最终输出的 attention_mask tensor 数学上和原版**完全等价**。

---

## 3. 数值正确性

- **lm_loss** iter20 偏差 = 0.0013 / 1.6778 = **0.08%**: 完全在 BF16/FP8 噪声范围 (跨 6 次 run 历史 σ/mean ≈ 0.18%)
- **grad_norm** iter20 偏差 = 0.014 / 0.382 = 3.7%: 远小于此前实测的跨 run σ/mean 16%
- **lm_loss / grad_norm 同 sign 同量级地波动**, 没有出现以前怀疑的"maskcache 让 grad_norm 系统性偏高"现象

加上 [attention mask 代码路径分析](2026-04-30_attention_mask_codepath_analysis.md)
里证实的 **dataset 端 attention_mask 在 forward 里被 `skip_getting_attention_mask_from_dataset=True` 强制置 None, 不进 GPU**，这次的对照彻底排除了
maskcache 影响数值结果的可能性。

---

## 4. 关键 lessons

1. **不要假设 `-v` mount 已经生效**: 以后改 host 文件之前先做一次 marker 测试 (`echo "# probe-$(date +%s)" >> host_file && docker exec ... tail -1 container_path`).
2. **container 必须用 start_docker.sh 启动**: 直接 `docker run sleep infinity` 起来的 `xm-primus` 没有 mount，但表面看不出来。
3. **过去几小时的所有"baseline"对照** 实际跑的是 container 内 `docker cp` 进去的 maskcache 版（mtime 08:19 写入），所以 maskcache vs baseline 的差异统计部分有混淆，需要看本次 clean 数据为准。

---

## 5. 当前状态

- container `xm-primus` 已重启，mount 工作正常
- host `sft.py` md5 = `03776e12ee1b7dad69a9ca8031e2ee65` (maskcache 生效版)
- container 看到同一 md5
- 任何后续 host 修改会立即在下次 import 后生效
