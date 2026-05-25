# 2026-05-13 19:30 CST  MMOE super-kernel × Primus / DSV3 4-layer e2e bring-up — 4 个静默 crash 全部根因定位

## 环境
- 主机: mi355-gpu-26 (AMD MI355X gfx950, 8 × 192 GB HBM3, XGMI 全互联)
- 容器: xiaoming-dev (podman)
- 工具链: ROCm 7.2 / PyTorch 2.10 nightly + flash-attn 2.8.3 + Transformer-Engine
- 训练栈: Primus + Megatron-LM (third_party/Megatron-LM @ 0.16.0rc0)
- 接入入口: `examples/primus/dsv3_4layer_mmoe.yaml` (PP=1 / EP=8 / TP=1, 4 layers, mock data, train_iters=20)
- patch 入口: `Primus/primus/backends/megatron/patches/moe_patches/mmoe_super_kernel_patches.py` (phase=`before_train`)
- forward 路径: `mmoe.megatron.MonolithMoELayer` → `mmoe.function.MonolithMoEFunction` → `csrc/fused_moe_super_kernel.hip`
- backward 路径: `MMOE_BACKWARD=eager`（autograd-aware 复现 forward）

## 问题
方案 B 接 Primus 后，DSV3 4-layer 真实训练 8-rank torchrun **全部 4 次失败都没有 Python traceback**（rank 1/4/6 任一 exitcode=1，其余 SIGTERM）。stderr.log 只有 FBGEMM 初始化噪声然后 `destroy_process_group` warning，从外面无法看到任何报错。

## 主要发现 / 结论

| # | 阶段 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | model build | rank 0 卡在 `transformer_config.py:1652` warning 后静默退出，其他 rank exitcode=1 | Primus 自身的 `topk_router_patches.py:195` 已经把 `moe_layer.MoELayer.__init__` 替换为 `patched_moelayer_init`；我们的 mmoe patch 又把 `sys.modules["...moe_layer"].MoELayer` 改成 `MonolithMoELayer`，导致原始 `MoELayer.__init__:172` 里的 `super(MoELayer, self).__init__(...)` 在运行期解析为 `super(MonolithMoELayer, self)`，循环回到 wrap 过的 `patched_moelayer_init`，**Python infinite recursion**（RecursionError 被 Primus `traceback.print_exc()` 写到 stderr 但被 torchrun `--redirects 3` 吞掉） | 只 patch `moe_module_specs.MoELayer` 一处（spec factory 的消费侧），**不要** rebinding `moe_layer.MoELayer`；保留原始 class identity 让 `super()` MRO 正常走 |
| 2 | forward, layer 1 | `AttributeError: 'TEColumnParallelGroupedLinear' object has no attribute 'weight'. Did you mean: 'weight0'?` | DSV3 走 `TEGroupedMLP`，权重存为 `weight0`/`weight1`/…/`weight{E_local-1}`（每 expert 一个 `[2F,H]` / `[H,F]` 参数），不是 fused 的单 `weight` | `_extract_grouped_mlp_weights` 增加 TE-grouped 分支：`torch.stack([linear.weight{i} for i in range(num_gemms)], dim=0)` 直接得到 `[E,2F,H]` / `[E,H,F]`（已经是 kernel 布局，不再 transpose）；`_ensure_transposed_weights` TE 分支不再依赖 `_version` 缓存（多 param、每次 forward 重 stack ≈ E_local 次 DMA） |
| 3 | forward, IPC bootstrap | `torch.AcceleratorError: HIP error: peer access is already enabled` 报在 `dist.all_gather_object` 里面的 `torch.zeros(...)` | NCCL/RCCL 在 init_process_group 时已经把 8 个 device 之间的 peer access 全部开了；我们 `hipDeviceEnablePeerAccess` 二次开启返回 `hipErrorPeerAccessAlreadyEnabled` 即使 C++ 侧忽略掉，**HIP 的 per-thread last-error slot 仍被污染**，下一次 `torch.zeros` 触发的同步检查把它当成自己的 crash | `enable_peer_access` 入口先 `(void)hipGetLastError();` 清空旧 error，loop 内每次 `hipDeviceEnablePeerAccess` 之后再 `(void)hipGetLastError();` drain。**幂等 + last-error 安全** |
| 4 | backward, eager replay | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | `_eager_moe_reference` 用了 `dist.all_to_all_single` 和 `out_recv.index_copy_(...)`。前者**不是 autograd-aware** —— 通信后的 `recv_h` 没 `grad_fn`，整个 expert 计算图断在 alltoall；后者 in-place 写入非 `requires_grad` 的 `out_recv` 也不接梯度 | 把 3 个 token-side alltoall 改成 `torch.distributed.nn.all_to_all_single`（autograd-aware；签名 `(output, input, ...)`）；把 expert 输出聚合改成 `torch.zeros(...).index_add(0, cat_idx, cat_y)` 这种 out-of-place + 可微的形式 |

修完后：8-GPU 8 rank torchrun 跑完 20 iterations `train_iters`，`Megatron pretrain execution completed.` / `Training completed.` / `Cleanup completed.` 全程无 NaN（Megatron 的 `check_for_nan_in_loss_and_grad=True` 没触发）；VRAM 占用 167 GB/rank（DSV3 4-layer + 3 MoE layer + MMOE workspace ≈ 1 GB/layer），属于健康区间。

## 详细分析

### 排障关键：拿到第一份 Python traceback

前 3 次失败 stderr 都只有 FBGEMM init noise + `destroy_process_group` warning，**没有任何 traceback**。逐步用以下办法剥离才看到 root cause：

1. **`sys.excepthook` wrapper** (`/tmp/run_wrapper.py`)
   - 用 `runpy.run_path` 包 `primus/cli/main.py`，wrapper 写自己的 excepthook + SystemExit traceback dump 到 `/tmp/mmoe_errlogs/rank{R}.err`
   - 结果：8 个 rank 全部 `SystemExit(1)` 来自 `primus/cli/main.py:188` —— 说明 Primus 已经 catch 到 `Exception`、调用了 `traceback.print_exc()`，然后 `raise SystemExit(1)`
   - 但 `traceback.print_exc()` 写的 stderr **没出现在 `--redirects 3` 捕获的 log 里**（loguru 没 hijack stderr，但 torchrun 的 fd-level redirect 在 worker 进程 fork 时拷贝了 fd 2 → log 文件；Primus 自己 `traceback.print_exc()` 的输出**理应**进 log 但实测没看到 —— 怀疑是 print 缓冲 + 进程组终止顺序）

2. **直接 patch Primus 的 fatal handler**
   - 在 `primus/cli/main.py:188` 的 `except Exception:` block 里，把 `traceback.format_exc()` 同时写一份到 `${PRIMUS_FATAL_DUMP_DIR}/rank${RANK}.err`
   - 这一改让 4 个 issue（recursion / TE weight / peer-access / autograd bwd）依次被定位

> **可移植结论**：torchrun `--redirects 3` 在 worker 异常退出场景下偶尔会丢 stderr，尤其当多 rank 同时往 stderr 写时；**永远在 launcher 主入口加一个 per-rank 文件 dump fallback**，不要相信 `--redirects` 捕获的内容是完整的。

### Issue 1: `super()` MRO 被 module-name 重绑破坏

Megatron 原始 `MoELayer.__init__` 是 Python 3 风格的 `super(MoELayer, self).__init__(...)`（不是裸 `super()`）。这种写法在运行期把 `MoELayer` 当成 module-level 名字解析。Primus 的 `topk_router_patches` 已经把 `moe_layer.MoELayer.__init__ = patched_moelayer_init` —— 这只改了**方法槽**，没改 class identity，所以 `super(MoELayer, ...)` 仍然解析为 MoELayer 父类 BaseMoELayer，没问题。

但我们的 `mmoe_super_kernel_patches.py` 还**进一步**把 `sys.modules["...moe_layer"].MoELayer = MonolithMoELayer`：
- 此时 `MonolithMoELayer.__init__` (`mmoe/megatron.py:130`) → `super().__init__()` → MRO 找到父类的 `__init__`，即 `moe_layer.MoELayer.__init__` = `patched_moelayer_init`
- `patched_moelayer_init` → `original_moelayer_init(self, ...)` = 原始 `MoELayer.__init__`
- 原始 `MoELayer.__init__:172` → `super(MoELayer, self).__init__(...)` —— 这里的 `MoELayer` 是 module 内的 global，已经被我们换成 `MonolithMoELayer` —— 所以解析为 `super(MonolithMoELayer, self).__init__(...)` —— **再次走到 patched_moelayer_init** —— **infinite recursion**。

栈深度上千；Primus 的 fatal dump 显示 stack 反复在 `mmoe/megatron.py:130 → topk_router_patches.py:193 → moe_layer.py:172 → topk_router_patches.py:193 → moe_layer.py:172 → ...` 循环。

**修复**：只 patch `moe_module_specs.MoELayer`（spec factory 在 `module=MoELayer` 时按 module global 解析），不动 `moe_layer.MoELayer`。原始 `MoELayer.__init__` 里的 `super(MoELayer, ...)` 仍然指向原始 class，正常下 MRO 到 BaseMoELayer，递归断开。

```python
# 错误（之前）：
target_modules = ["megatron.core.transformer.moe.moe_layer",
                  "megatron.core.models.gpt.moe_module_specs"]

# 正确：
target_modules = ["megatron.core.models.gpt.moe_module_specs"]
```

### Issue 2: `TEColumnParallelGroupedLinear` 权重布局

TE 的 `GroupedLinear` 每 expert 一个 `weight{i}` Parameter（与 fp8 quant scale 一一对应），没有 fused `weight` 属性。DSV3 默认走 `TEGroupedMLP`，我们之前只支持 fused `weight1`/`weight2`（legacy `GroupedMLP`）和 `linear_fc{1,2}.weight`（plain TELinear），命中不到。

**修复**：在 `_extract_grouped_mlp_weights` 加 TE-grouped 分支：

```python
if hasattr(fc1, "weight0"):
    num = getattr(fc1, "num_gemms", _discover_num(fc1))
    w1 = torch.stack([getattr(fc1, f"weight{i}") for i in range(num)], dim=0)
    w2 = torch.stack([getattr(fc2, f"weight{i}") for i in range(num)], dim=0)
    return w1, w2  # shape [E,2F,H] / [E,H,F] — already kernel-layout
```

stack 后的形状直接是 kernel 布局，**不再 transpose**。

代价：每次 forward 重 stack（≈ E_local 次 DMA copy），DSV3 EP=8 时 E_local=128/8=16，单层 stack 量 ~1.4 GB / layer / forward，相对 expert GEMM 流量可忽略。后续 P1 可以让 stack 落到独立 stream + L2 prefetch。

### Issue 3: HIP last-error slot 污染

容易踩、文档很少的坑。`hipDeviceEnablePeerAccess` 是同步 host call，return value 是 enum；**即使你拿到 return value 并按 `hipErrorPeerAccessAlreadyEnabled` 处理掉，HIP 仍然把它写进 per-thread last-error**。下一次任何同步 HIP 调用（`hipMalloc`、`hipMemset`、`torch.zeros` 走的 `hipMallocAsync`...）都会调用 `hipGetLastError()` 然后把这个 stale "already enabled" 当成**自己**的错误抛出来 —— Python 看到的是 `torch.zeros: HIP error: peer access is already enabled`，stack 完全无关。

**修复**：bindings.hip `enable_peer_access` 入口和 loop 内主动 `(void)hipGetLastError()` drain：

```cpp
void enable_peer_access(int rank, std::vector<int> peers) {
    MMOE_HIP_CHECK(hipSetDevice(rank));
    (void)hipGetLastError();                    // drain any pre-existing error
    for (int peer : peers) {
        if (peer == rank) continue;
        ...
        hipError_t err = hipDeviceEnablePeerAccess(peer, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) { ... }
        (void)hipGetLastError();                // drain benign AlreadyEnabled
    }
}
```

这条同时适用于**所有**「benign-return-code-但污染-last-error」的 HIP API，比如 `hipMemcpyPeer` 跨 device 时的某些状态。

### Issue 4: eager backward replay 必须用 autograd-aware collectives

`MMOE_BACKWARD=eager` 复现 forward 让 autograd 帮我们求 grad。但 `dist.all_to_all_single` **不是** autograd-aware —— 它把 input copy 到 output buffer 完事，**没注册任何 `grad_fn`**。结果 `recv_h` 是没 `grad_fn` 的 fresh tensor，整张计算图在 alltoall 处断开，`autograd.grad(y_ref, inputs=(h, pr, w1, w2))` 第一个 `outputs=y_ref` 不 require_grad → `element 0 of tensors does not require grad and does not have a grad_fn`。

**修复**：

```python
import torch.distributed.nn as dist_nn  # autograd-aware

# 旧：dist.all_to_all_single(recv_h, sorted_h, ...)
# 新：
recv_h = torch.empty(total_recv, H, dtype=..., device=...)
recv_h = dist_nn.all_to_all_single(recv_h, sorted_h.contiguous(), ...)
```

签名差异：`dist.all_to_all_single` 是 `(output, input, ...)` 原地写入并返回 `Work`；`dist_nn.all_to_all_single` 也是 `(output, input, ...)` 但**返回带 grad_fn 的 output tensor**，所以要赋值回去。

同时把 expert 输出聚合从 `out_recv.index_copy_(0, idx, y_e)`（in-place、不可微）改成 `torch.zeros(...).index_add(0, cat_idx, cat_y)`（out-of-place、可微）。

token expert id 是 int32 不需要 grad，仍走 `dist.all_to_all_single`；只有 activation 和 router weight 才需要 `dist_nn`。

## 实验数据

```
container: xiaoming-dev @ mi355-gpu-26
config:    examples/primus/dsv3_4layer_mmoe.yaml (PP=1 EP=8 TP=1, 4 layers, 3 MoE layers)
env:       MMOE_OFFLOAD_ARCH=gfx950 PYTORCH_ROCM_ARCH=gfx950 MMOE_BACKWARD=eager
launcher:  torchrun --standalone --nproc_per_node=8 primus/cli/main.py train pretrain ...

iter 20/20:  Megatron pretrain execution completed.
NaN check :  green (check_for_nan_in_loss_and_grad=True 未触发)
VRAM      :  rank0..7 167.17 ~ 168.09 GB used  (MI355X 192 GB headroom 24 GB)
wall      :  total ~71 s (model build 41s + 20 iter 30s 含 warmup)
```

## 下一步 / 建议

| 优先级 | 方向 | 说明 |
|---|---|---|
| P0 | 跑 loss baseline 对比 | 把同一 4-layer 配置在**不带** `use_mmoe_super_kernel` 的情况下也跑 20 iter，对比 iter-20 loss 是否在合理 ε 内（mock-data + 同 seed 期望 bit-similar 难，**关注 loss 趋势是否一致 + grad-norm 是否健康**） |
| P0 | 实现 `_backward_decomposed` | eager replay 当前是 forward 的 2-3 倍 wall（重跑了一遍 alltoall + 全 expert GEMM）；用 ws 里保留的 `permuted_input` / `fc1_gate_up_save` 拼 hand-written bwd，预期 backward ≈ 1× forward |
| P1 | 把 fatal-dump 改成可选 + 写到 exp output 目录 | 当前修改的 `primus/cli/main.py:188` 提交前要让 dump 默认关闭、只有 `PRIMUS_FATAL_DUMP_DIR` 显式设置才写，避免污染 Primus 主仓 |
| P1 | TE-grouped weight stack 落到独立 stream + L2 prefetch | E_local=16 时单层 1.4 GB DMA per forward 当前阻塞了 expert GEMM 启动；并发上去后影响小，但训练 wall 现在还是 forward 占主导 |
| P1 | 支持 `T_local` 在 micro-batch 间漂移导致的 workspace 重新 alloc | 当前 `_ensure_workspace` 会在 T_local 超过 max_recv 时直接 `ws.free() + 重新 allocate` —— 对于 mock data 不会触发，但 DPO/SFT 真实 dataset 可能会 |
| P2 | 多 micro-batch / 多层 workspace 复用 | 当前每个 MoE layer 维护独立 workspace，3 MoE layer × ~1 GB = 3 GB；可以全模型共享一份 |

## 相关文件

- 训练入口: `examples/primus/dsv3_4layer_mmoe.yaml`
- 启动脚本: `examples/primus/run_dsv3_mmoe.sh`
- Primus patch: `3rd/Primus/primus/backends/megatron/patches/moe_patches/mmoe_super_kernel_patches.py`
- Megatron 集成: `python/mmoe/megatron.py`
- Autograd Fn: `python/mmoe/function.py`
- IPC workspace: `python/mmoe/workspace.py`
- HIP bindings: `python/mmoe/_csrc/bindings.hip`
- Super-kernel 改造: `csrc/fused_moe_super_kernel.hip`（已落地 `fc1_gate_up_save` 可选 save，eager bwd 暂时没用到，留给 decomposed bwd）
- 调试用 fatal dump 入口: `3rd/Primus/primus/cli/main.py:188` (PRIMUS_FATAL_DUMP_DIR)
