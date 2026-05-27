# 2026-05-25 22:00 UP — FIFO-slot barrier fixes DeepEP intranode hang on AMD/XGMI

> **结论**: M1e DeepEP 端口的间歇性 hang 根因是 vendor 的 NVIDIA-style `barrier_block` 在 AMD MI355X 上不 race-free。换成 ROCm/DeepEP fork 的 **FIFO-slot barrier (`barrier_device` + `move_fifo_slots`)** 后, 12 个 fresh process × 20 iter + 单 process 6000 iter (含 head wrap-around) 全 PASS, 旧版 ~50% 通过率 → 100%。`PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE=1` 这个 workaround 也不再需要。

## 现象

M1e.3 落 `csrc/dispatch_deepep.hip` 后跑 `tests/test_deepep_dispatch.py` (EP=2, B=32, K=4, H=256, E_local=4) 间歇性 hang —— 单次 dispatch 第一帧大概 ~50% 概率卡在 `torch.cuda.synchronize()`, 没有 `DeepEP timeout` print, 30 s 内无 SIGABRT, 只能靠外部 timeout 杀。

打 `trace()` (在 `dispatch_deepep::launch` 每个 sub-kernel 前后 `hipStreamSynchronize` + `fprintf`) 后看清楚: **失败时两边都进 `notify_dispatch`, 其中一 rank 同步过去拿到结果, 另一 rank 卡死在 kernel 里**。比如:

```
rank 0: after layout / before notify_dispatch
rank 1: after layout / before notify_dispatch
rank 0: after notify_dispatch / before dispatch   ← rank 0 sync 过了
                                                  ← rank 1 永远停在这
```

如果是普通的 cross-rank deadlock, 两 rank 都该卡。**一边过、一边死**就是单 slot atomic ordering 问题的指纹。

## 根因

vendor 的 `csrc/deep_ep/utils.cuh::barrier_block<R>` 是上游 NVIDIA DeepEP main branch 的实现:

```cpp
// 所有 barrier 调用都用同一片 R 个 int 的 barrier_signal 区
if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank]      + thread_id, FT);
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank,      FT);
}
while (true) {
    auto value = ld_volatile_global(barrier_signal_ptrs[rank] + thread_id);
    if (__all_sync(value <= 0)) break;
    if (clock64() - start > TIMEOUT) trap();
}
```

`notify_dispatch` 里连续 call 3 次同一个 slot。NVIDIA H100/B200 上 `atomicAdd_system` / `atomicSub_system` 走 NVLink 是 strong ordered 的, 同 slot reuse 没事。**MI355X / CDNA4 XGMI 不保证背靠背的 system-scope atomic 到同一地址的顺序**, 上一次 barrier 的 ± 还没全部 land, 下一次 barrier 的 ± 就盖上去, spin loop 偶尔会看到 phantom `value == 0` 然后某一 rank 提前退出, 留下没人去 sub 的剩余 +FT 把另一 rank 永远 deadlock 在 `while`。

ROCm/DeepEP fork (commit 577efe82, `csrc/kernels/utils.cuh`) 的解法是 **task FIFO**:

```cpp
template <int kNumRanks>
__forceinline__ __device__ void barrier_device(int **task_fifo_ptrs, int head, int rank) {
    if (thread_id < kNumRanks) {
        atomicAdd_system(task_fifo_ptrs[rank]      + head + thread_id, FT);
        memory_fence();
        atomicSub_system(task_fifo_ptrs[thread_id] + head + rank,      FT);
    }
    // spin on (task_fifo_ptrs[rank] + head + thread_id) == 0, with timeout
}

template <int kNumRanks>
__forceinline__ __device__ void move_fifo_slots(int &head) {
    head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}
```

关键: 每个 barrier 用 `head` 指向的一段全新 R 个 slot, 调用之间 `move_fifo_slots` 把 head 推进 R。NUM_MAX_FIFO_SLOTS = 32768 (`csrc/kernels/configs.cuh`)。host 端在 `deep_ep.cpp::Buffer::move_fifo_slots(int num_slots)` 把 `head` 模 NUM_MAX_FIFO_SLOTS, 每个 kernel call 推进 `num_ranks * num_barriers_in_kernel` 个 slot。

因为没人会 reuse 旧 slot, 没办法看到上次 barrier 残留的 phantom 0, AMD 弱 ordering 就 race 不出来。

## 改动 (RocMoE)

落在 RocMoE 的端口跟 ROCm/DeepEP fork 一样:

| 文件 | 改动 |
|---|---|
| `csrc/include/rocmoe/deep_ep/configs.h` | `NUM_MAX_FIFO_SLOTS = 32768` |
| `csrc/deep_ep/utils.cuh` | `barrier_block` → `barrier_device<R>(task_fifo_ptrs, head, rank)` + `move_fifo_slots<R>(head&)`, spin condition 改 `value == 0` (更严), `memory_fence()` 加在 add/sub 之间 |
| `csrc/deep_ep/intranode.hip` | `notify_dispatch` / `cached_notify_dispatch` / `cached_notify_combine` 全部 plumb `int head` 参数, 每次 barrier 后 `move_fifo_slots<R>(head)` (`notify_dispatch` 用 3 次, `cached_notify_dispatch` / `cached_notify_combine` 各 2 次) |
| `csrc/include/rocmoe/deep_ep/api.h` | 3 个 launcher 的签名加 `int head` |
| `csrc/include/rocmoe/dispatch.h` | `DispatchArgs` 加 `int barrier_head`, 新增 `kNotifyDispatchBarriers = 3` |
| `csrc/include/rocmoe/workspace.h` | `barrier_signal_ptrs` → `task_fifo_ptrs` (rename + 含义换) |
| `csrc/include/rocmoe/moe_config.h` | `barrier_buffer_bytes` → `task_fifo_bytes = NUM_MAX_FIFO_SLOTS * sizeof(int)` (128 KB) |
| `csrc/moe_config.cpp` | `static_assert(NUM_MAX_FIFO_SLOTS % kMaxRanks == 0)` 保 wrap 干净 |
| `csrc/dispatch_deepep.hip` | 把 `args.barrier_head` 喂给 `notify_dispatch` |
| `python/rocmoe/_C.cpp` | `PyWorkspace` 加 `barrier_head_` + `current_barrier_head()` + `advance_barrier_head(num_barriers)`, 在 `py_dispatch_deepep` 里 launch 后自动调用 `ws.advance_barrier_head(kNotifyDispatchBarriers)`, 用户不感知; `reset_barrier()` → `reset_counters()` (FIFO 不需要 reset) |
| `tests/test_deepep_dispatch.py` | 跟 API rename 对齐, 加 `--repeat N` 验证多 iter 稳定性 |

每个 dispatch 调用消耗 `R * 3 = 6` 个 slot (R=2 时), 一圈 `32768 / 6 = 5461` 次 dispatch。

## 验证

`build/python` 现版:

```
small  (B=32  K=4  H=256  E_local=4)  × 50 iter / process × 12 fresh process    → 12/12 ALL PASS  (7.5s / process)
medium (B=128 K=8  H=1024 E_local=8)  × 30 iter / process                       → ALL PASS
large  (B=256 K=8  H=2048 E_local=16) × 30 iter / process                       → ALL PASS
wrap   (B=32  K=4  H=256  E_local=4)  × 6000 iter / process (head 跨过 32768)    → ALL PASS, head_after=3226 = 5999*6 mod 32768
```

`PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE` 不再设, 默认 cheap_fence = on 也稳。

旧版同 shape 跑 10 次大约 5-6 次卡 (Process got signal: 15), 30 秒 timeout 也来不及看 DeepEP kernel 的 timeout print。改完是 0/N。

## 不要做的事

1. **不要尝试在 barrier 之前 `hipMemsetAsync(barrier_signal_ptrs, 0)` 强制清零**. 之前花了 ~1 小时调 `reset_barrier()` 各种位置, 全部 hang 模式不变 —— memset 跟 atomic 之间也有 ordering 问题, 而且就算 memset 干净, 下一个 barrier 的 in-flight 旧 atomic 还是会污染新 slot。FIFO 才是正解。
2. **不要把 `barrier_signal_ptrs` 切成 R 段独立分配**. 试过 (单 R*sizeof(int) 一个 hipMalloc + 单独 hipIpcGetMemHandle), 偶尔 hang 还会换成更严重的 lazy-peer-access deadlock (两个 `hipIpcOpenMemHandle` 背靠背调到同一 peer device 在 ROCm 6.x 不 re-entrant)。combined `[ipc | task_fifo]` 单 hipExtMallocWithFlags(uncached) + 单 IPC handle 才稳。
3. **不要去掉 `attach_peers` 里那个 `int probe; hipMemcpy(&probe, peer_base, sizeof(int), DtoH)` 同步 probe**. 它 force 把 lazy peer-access 路由在 stream 上提前建好, 避免第一帧 dispatch 跟 IPC 路由 setup 抢资源 (做完 FIFO fix 也保留, ablation 显示 5% 概率的 first-frame race 由它 cover)。

## 下一步 (M1e.6)

1. 把现在 `tests/test_deepep_dispatch.py` 的 `--repeat` 多 shape 跑法包成 ctest entry
2. 写 `benchmarks/bench_deepep_dispatch.py` (host-only profile + per-shape p50/p90), 跟旧 `bench_dispatch.py` 对得齐
3. M1f 装 combine kernel 时, `cached_notify_combine` 也是 2 个 barrier, host 端要 `advance_barrier_head(2)`. 已经在 `dispatch::kNotifyDispatchBarriers` 旁边留好 hook。

## 参考

- ROCm DeepEP `csrc/kernels/utils.cuh` `barrier_device` / `move_fifo_slots`
- ROCm DeepEP `csrc/kernels/configs.cuh` `NUM_MAX_FIFO_SLOTS = 32768`
- ROCm DeepEP `csrc/deep_ep.cpp` `Buffer::move_fifo_slots(int)`, `head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS`
- AMD CDNA 弱 ordering wrt repeated system-scope atomic to same address: 之前在 `slab/knowledge/kernels/` 没专门归档, 这次的 case 算第一份 reproducer
