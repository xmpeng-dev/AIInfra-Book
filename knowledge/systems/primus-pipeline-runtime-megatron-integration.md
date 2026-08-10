# Primus 流水线 runtime 如何接入 Megatron（完整走通一个例子）

> 依据 `~/workspace/Primus` HEAD `11400635` 的实际代码走读。配置示例为 PP=4 / VPP=1 / 8 micro-batch / `pp_algorithm: "1f1b"`。
> 战略与规划上下文见 `notes/primus-moe/2026-08-04_1049_primus-runtime-design-and-plan.md`。

---

## 先搞清楚：Primus 有三条流水线路径，本文讲的是第三条

读代码前必须分清，否则会走错栈：

| 路径 | 开关 | 代码位置 | 行数 | 谁在用 |
|---|---|---|---|---|
| **原生 Megatron schedules** | 两个开关都不开（默认） | 上游 | — | 大部分 recipe，含 DeepSeek-V3 671B（PP16 / VPP2） |
| **legacy `zerobubble`** | `--patch_zero_bubble True` | `backends/megatron/core/pipeline_parallel/zerobubble/` | 7,542 | **生产路径**：`examples/customer_package/run_qwen3_{235b_a22b,30b_a3b}_pretrain_mi355x.sh` feature case 7，默认 `PP_STRATEGY="zbv"` |
| **`primuspipe` + core**（本文对象） | `patch_primus_pipeline: true` | `core/pipeline_parallel/` + `backends/megatron/.../primuspipe/` | 2,277 + 1,085 | **仅 2 个测试 yaml**（`tests/trainer/test_megatron_trainer_zbv_fp8.yaml`、`test_megatron_trainer_zero_bubble.yaml`） |

**本文讲第三条，因为它是目标架构（plan/execute 分离、后端无关内核）——但要注意它目前不在生产路径上。** 生产上跑的是第二条，即绑定 megatron 的 legacy 栈。

两个佐证迁移意图存在但停在测试层的细节：

- `tests/trainer/test_megatron_trainer_zero_bubble.yaml` 里 `patch_zero_bubble: true` 被**注释掉**、替换为 `patch_primus_pipeline: true`。
- 两套栈的策略名已重叠到用户配置面：`v-half` / `v-min` 在 primuspipe 是 `pp_algorithm` 的取值，在 legacy 是 `zero_bubble_v_schedule_mem_setup: half|min`。

`primuspipe` 路径把 Megatron 的**整个 step 执行权**接了过来：Megatron 不再决定 micro-batch 的顺序和通信时机，它只提供模型的前反向计算。本文用一个 PP=4 的具体配置，把从 yaml 到 kernel 的五层调用链完整走一遍。

**核心结构：plan / execute 分离。**

```
配置 (yaml)
  ↓
planner   PipelineScheduleAlgo.generate_schedule_table()  →  plan: list[list[SchedulerNode]]
  ↓                                                          （后端无关，可序列化）
adapter   PrimusPipelineParallelLauncher.run()            →  给每个 node 填 args / meta
  ↓
executor  ScheduleRunner.run(table, rank)                 →  按 node.func_type 分派
  ↓
handler   megatron_fwd_handler / default_wgrad_handler …  →  回调 Megatron 的 forward_step
```

上三层与后端无关（住在 `primus/core/`），下两层是后端适配（住在 `primus/backends/megatron/`）。

---

## Step 0：配置

```yaml
# primus/configs/modules/megatron/primus_pipeline.yaml
patch_primus_pipeline: true          # 打开执行权移交
pp_algorithm: "1f1b"                 # 选调度算法
communication_method: "async_p2p"
debug_scheduler_table: true          # 打印 plan，调试时很有用
offload: false
```

其余 Megatron 参数（`pipeline_model_parallel_size: 4`、模型定义、数据路径）**完全不变**——这是 drop-in 兼容性的体现。

可选算法由 `produce_schedule_instance` 的注册表决定：`1f1b`、`1f1b-interleaved`、`zero-bubble`、`zero-bubble-heuristic`、`zbv-formatted`、`v-half`、`v-min`。

---

## Step 1：绑定——把执行权交出来

`patch_primus_pipeline: true` 触发一个 patch，改写两处模块绑定：

```39:47:primus/backends/megatron/patches/parallelism/schedule_patches.py
    ori_pp.get_forward_backward_func = get_primus_pipeline_parallel_fwd_backward_func
    ...
    megatron_training.get_forward_backward_func = get_primus_pipeline_parallel_fwd_backward_func
```

被换上去的东西只有两行：

```120:121:primus/backends/megatron/core/pipeline_parallel/schedules.py
def get_primus_pipeline_parallel_fwd_backward_func():
    return PrimusPipelineParallelLauncher().run
```

此后 Megatron 的 `train_step` 调 `get_forward_backward_func()` 拿到的就是 Primus 的 launcher。**这是整个接入的唯一入口**——其余一切都发生在这个函数内部。

---

## Step 2：planner 产出 plan

`Schedule1F1B.generate_schedule_table()` 按 rank 生成 node 列表。它只依赖 `(pp_size, vpp_size, micro_batches)`，不碰任何 torch 对象，纯粹是排期计算。

节点表示法 `(TYPE|mini_batch|chunk)`，取自 `SchedulerNode.__str__`。PP=4、8 micro-batch 下的实际结果：

| rank | warmup | steady | cooldown |
|---|---|---|---|
| **0** | `(F\|0\|0) (SF\|0\|0) (F\|1\|0) (SF\|1\|0) (F\|2\|0) (SF\|2\|0)` | `(F\|3\|0) (SF\|3\|0) (RB\|0\|0) (BW\|0\|0)` … 至 `(F\|7\|0) (SF\|7\|0) (RB\|4\|0) (BW\|4\|0)` | `(RB\|5\|0) (BW\|5\|0) (RB\|6\|0) (BW\|6\|0) (RB\|7\|0) (BW\|7\|0)` |
| **1** | `(RF\|0\|0) (F\|0\|0) (SF\|0\|0) (RF\|1\|0) (F\|1\|0) (SF\|1\|0)` | `(RF\|2\|0) (F\|2\|0) (SF\|2\|0) (RB\|0\|0) (BW\|0\|0) (SB\|0\|0)` … | `(RB\|6\|0) (BW\|6\|0) (SB\|6\|0) (RB\|7\|0) (BW\|7\|0) (SB\|7\|0)` |
| **3** | 无（`warm_up = pp_size - rank - 1 = 0`） | `(RF\|0\|0) (F\|0\|0) (BW\|0\|0) (SB\|0\|0)` … 至 `(RF\|7\|0) (F\|7\|0) (BW\|7\|0) (SB\|7\|0)` | 无 |

两个值得注意的点：

- **通信节点是 plan 里的一等公民**，不是隐式副作用。`generate_send_recv_nodes()` 依据 `direction_map()` 给出的邻接关系决定要不要插 `RF`/`SF`/`RB`/`SB`；rank 0 没有上游所以没有 `RF`，rank 3 没有下游所以没有 `SF`。这是后续能做通信编排与 graph 捕获的前提。
- **`1f1b` 用的是 `BW`（B 与 W 合并），不产生独立的 `W` 节点。** 只有 zero-bubble 系列才会把 `W` 拆出来单独排期——所以 `default_wgrad_handler` 在本例中不会被调用（见 Step 5）。

`debug_scheduler_table: true` 会调 `print_schedule_table()` 打印上表。

---

## Step 3：adapter 把 plan 变成可执行的

plan 是纯数据，不含 torch 对象。**把后端上下文注入 node 是 adapter 唯一的实质职责**：

```289:309:primus/backends/megatron/core/pipeline_parallel/primuspipe/pipeline_launcher.py
            if node.func_type == FuncType.F:
                node.args["forward_step_func"] = forward_step_func
                node.args["data_iterator"] = data_iterator
                node.args["models"] = model
                ...
                node.args["recv_tensor_shapes"] = recv_tensor_shapes
            elif node.func_type in [FuncType.B, FuncType.BW]:
                node.args["model_type"] = model_type
                node.args["send_tensor_shapes"] = send_tensor_shapes
            elif node.func_type in [FuncType.RF, FuncType.RB]:
                node.args["recv_tensor_shapes"] = recv_tensor_shapes
                node.args["dtype"] = config.pipeline_dtype
                node.args["pp_group"] = pg_collection.pp
```

`meta` 里放 `pp_size` / `vpp_size` / `pp_rank` / `last_pp_stage_rank`，`args` 里放该 op 类型需要的后端对象。launcher 同时负责 Megatron 侧的常规事务：process group 收集、`no_sync` 上下文、tensor shape 推导。

---

## Step 4：executor 分派

执行器全文 43 行，逻辑就是遍历本 rank 的 node 并查表分派：

```24:35:primus/core/pipeline_parallel/scheduler/scheduler.py
    def run(self, scheduler_table, rank: int):
        for idx, node in enumerate(scheduler_table[rank]):
            if node.args is not None and "combined_group" in node.args:
                func = self.handle_func_dict[FuncType.FB]
                func(node, idx, scheduler_table[rank])
            else:
                if self.pre_process_func is not None:
                    self.pre_process_func(node, idx, scheduler_table[rank])
                func = self.handle_func_dict[node.func_type]
                func(node, idx, scheduler_table[rank])
```

分派表就是后端提供的那张 dict——**11 个 op 里 4 个直接用 core 的默认实现**：

```24:36:primus/backends/megatron/core/pipeline_parallel/primuspipe/handlers/__init__.py
megatron_primuspipe_handler_dict = {
    FuncType.F: megatron_fwd_handler,
    FuncType.B: megatron_bwd_handler,
    FuncType.W: default_wgrad_handler,
    FuncType.O: default_offload_handler,
    FuncType.R: default_reload_handler,
    FuncType.BW: megatron_bwd_handler,
    FuncType.SF: batch_p2p_communication_handler,
    FuncType.SB: batch_p2p_communication_handler,
    FuncType.RF: batch_p2p_communication_handler,
    FuncType.RB: batch_p2p_communication_handler,
    FuncType.FB: megatron_combined_fwd_bkwd_handler,
}
```

---

## Step 5：handler 回调 Megatron

以 `(F|3|0)` 为例。handler 做三件事：**取输入、调 Megatron、把输出写回 node**。

**取输入**——不经过独立的 buffer manager，而是在 plan 里回溯找到对应的 recv 节点，读它的 `recv_buffers`：

```47:47:primus/backends/megatron/core/pipeline_parallel/primuspipe/handlers/fwd_handler.py
    idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RF])
```

```68:72:primus/backends/megatron/core/pipeline_parallel/primuspipe/handlers/fwd_handler.py
    input_tensors = (
        [None] * len(node.args["recv_tensor_shapes"])
        if idx is None
        else scheduler_table[idx].args["recv_buffers"]
    )
```

**plan 同时充当数据流图**——node 之间通过 `args` 传张量。这是 executor 能保持 43 行的原因。

**调 Megatron**——`forward_step` 是 Megatron 自己的函数，模型计算完全在上游代码里发生：

```75:80:primus/backends/megatron/core/pipeline_parallel/primuspipe/handlers/fwd_handler.py
        forward_step_func = forward_step
        kwargs["model"] = node.args["models"][node.chunk]
        kwargs["input_tensor"] = input_tensors
        kwargs["vp_stage"] = node.chunk
        kwargs["cp_group_size"] = node.args["cp_group_size"]
        kwargs["is_last_stage"] = is_last_stage
```

**写回输出**给下游节点消费：

```106:108:primus/backends/megatron/core/pipeline_parallel/primuspipe/handlers/fwd_handler.py
    node.args["total_num_tokens"] += num_tokens
    node.args["outputs"] = outputs if isinstance(outputs, list) else [outputs]
    node.args["inputs"] = input_tensors
```

### 延迟 wgrad：`W` 节点的机制

`W` 是 zero-bubble 的核心——把权重梯度计算从反向里拆出来延后执行。实现方式是让反向只把 wgrad 闭包塞进缓存，等 plan 走到 `W` 节点再冲刷：

```71:78:primus/core/pipeline_parallel/handler/wgrad_handler.py
def default_wgrad_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    cal_stored_grad_func = WGRAD_RUNNING_CACHE.flush
    if get_args().dump_pp_data:
        cal_stored_grad_func = fwd_bwd_wrapper(
            WGRAD_RUNNING_CACHE.flush, "wgrad", minibatch=node.mini_batch, chunk=node.chunk
        )
    cal_stored_grad_func(node.mini_batch, node.chunk)
```

`WGradRunningCache.append()` 有一个重要的降级行为：若当前没有设定 minibatch/chunk（即不在延迟模式），它**立即执行** wgrad。这让同一套算子代码在延迟与非延迟调度下都能工作。

本例用 `1f1b`（`BW` 合并），所以 `W` 不出现；换成 `zero-bubble` 就会看到独立的 `W` 节点。

---

## Step 6：收尾交回 Megatron

executor 返回后，launcher 做梯度收尾并把 loss 交回：

```324:346:primus/backends/megatron/core/pipeline_parallel/primuspipe/pipeline_launcher.py
        if config.finalize_model_grads_func is not None and not forward_only:
            finish_embedding_wgrad_compute(...)
            config.finalize_model_grads_func(
                model,
                total_num_tokens if config.calculate_per_token_loss else None,
                pg_collection=pg_collection,
                force_all_reduce=force_all_reduce,
            )
        assert WGradRunningCache.is_empty(), "WGradRunningCache is not empty"
        return self.forward_data_store
```

`finalize_model_grads_func` 是 Megatron 的，DP 规约与 optimizer step 仍归 Megatron。**控制流是 Megatron → Primus → Megatron 的三明治。**

---

## 接一个新后端需要写什么

对照上面的分层，新后端只需三样东西：

| # | 交付物 | 参考实现 | 量级 |
|---|---|---|---|
| 1 | 一张 `{FuncType: callable}` handler dict | `primuspipe/handlers/__init__.py` | 36 行 |
| 2 | 后端专属 handler（fwd / bwd / combined / p2p） | `primuspipe/handlers/*.py` | 约 700 行 |
| 3 | 一个 launcher：注入 args/meta + 后端收尾 | `pipeline_launcher.py` | 346 行 |

调度算法、IR、offload/wgrad 默认 handler 全部复用 core，**不需要重写一行**。这是「跨后端复用」的具体含义。

---

## 注意事项与当前的层泄漏

- **`core` 尚未真正后端无关。** `primus/core/pipeline_parallel/handler/wgrad_handler.py` 仍 `from megatron.training.global_vars import get_args`，并 import 了 megatron 侧的 `pp_visualizer`。16 个文件里 1 个有此问题，是 P0 的清理项。
- **两套 IR 并存，且 `R` 同名不同义。** core 的 `FuncType.R` 是 `RELOAD`（offload 回载）；`backends/megatron/.../zerobubble/scheduler/graph.py` 里的 `R` 是 recompute（被 `is_computation()` 判定为计算类）。读代码时务必确认在哪套 IR 下。
- **`primuspipe` 与 `zerobubble` 是两条独立路径，且生产在后者上。** 由 `patch_primus_pipeline` 与 `patch_zero_bubble` 分别开启，`megatron.pp.schedule` patch 的 `condition` 里两者互斥。功能覆盖以 `zerobubble` 更全（自动求解器、通信规划、细粒度 offload），所以本文描述的干净架构**尚未经过生产规模验证**——改动它之前不要假设有生产数据兜底。
- **plan 的双消费者性质要保住。** `primus/core/projection/` 的 simulator 复用同一套 IR 与算法类做性能预测；任何改动 IR 的重构都要同时验证 Projection 精度不退化。

---

## 相关

- 规划：`notes/primus-moe/2026-08-04_1049_primus-runtime-design-and-plan.md`
- 战略：`notes/primus-moe/2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md`
- Megatron 侧 MoE 实现要点：`papers/megatron-core-moe.md`
