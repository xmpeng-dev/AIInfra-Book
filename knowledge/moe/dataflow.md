# MoE 详细计算过程与数据流

> **定位：** 从工程实现角度，完整拆解一个 MoE Transformer Block 的前向、反向、通信和数据流。  
> **读者：** 需要自己实现或优化 MoE 训练/推理系统的工程同学。  
> **约定：** 以下默认以 Decoder-only LLM 的一个 MoE FFN 层为核心。

---

## 1. 符号与配置

```text
B: batch size
S: sequence length
T: token 数（T = B * S）
H: hidden size
F: expert FFN 中间维度
E: expert 总数
K: 每个 token 激活 expert 数（Top-K）
P: EP 组内 GPU 数
dtype_bytes: 数据类型字节数（bf16/fp16 = 2）
```

### 示例配置（用于后续具体数据流）

```text
B = 2, S = 8  => T = 16
H = 4096
F = 14336
E = 8
K = 2
P = 4（4 张 GPU 做 Expert Parallel）

专家放置：
GPU0: Expert 0, 1
GPU1: Expert 2, 3
GPU2: Expert 4, 5
GPU3: Expert 6, 7
```

---

## 2. 一个 MoE Block 的完整前向流程

## 2.1 输入到 Gate（路由打分）

输入激活：

```text
X: [B, S, H] = [2, 8, 4096]
展平后 X_flat: [T, H] = [16, 4096]
```

路由 logits：

```text
W_gate: [H, E] = [4096, 8]
logits = X_flat @ W_gate -> [T, E] = [16, 8]
probs = softmax(logits, dim=-1) -> [16, 8]
```

Top-K 选择：

```text
topk_idx: [16, 2]     # 每个 token 选 2 个 expert
topk_w:   [16, 2]     # 对应权重（可再归一化）
```

---

## 2.2 Token 路由映射（核心数据重排）

假设路由结果（仅示意）：

```text
t0 -> (e0, e3)
t1 -> (e1, e5)
t2 -> (e0, e6)
t3 -> (e2, e4)
t4 -> (e0, e2)
t5 -> (e3, e7)
t6 -> (e1, e4)
t7 -> (e5, e6)
t8 -> (e2, e7)
t9 -> (e0, e3)
t10 -> (e1, e5)
t11 -> (e4, e6)
t12 -> (e3, e6)
t13 -> (e1, e2)
t14 -> (e0, e7)
t15 -> (e2, e5)
```

这一步会生成三类索引元数据：

```text
1) token -> (expert_id, weight) 列表
2) 每个 expert 对应 token 列表（inverted index）
3) A2A send_counts / recv_counts（每卡发给每卡多少 token）
```

---

## 2.3 Dispatch All-to-All（第一次跨卡通信）

目标：把 token 副本送到目标 expert 所在 GPU。

注意：Top-K 路由意味着每个 token 会复制 K 份进入 expert 计算分支。

### 2.3.1 本示例中的 expert 负载

按上面映射统计：

```text
E0: t0 t2 t4 t9 t14              -> 5
E1: t1 t6 t10 t13                -> 4
E2: t3 t4 t8 t13 t15             -> 5
E3: t0 t5 t9 t12                 -> 4
E4: t3 t6 t11                    -> 3
E5: t1 t7 t10 t15                -> 4
E6: t2 t7 t11 t12                -> 4
E7: t5 t8 t14                    -> 3
总计 32 条 expert-token 分配（= T * K = 16 * 2）
```

每卡最终接收的 token 条数（按专家归属）：

```text
GPU0 (E0,E1): 5 + 4 = 9
GPU1 (E2,E3): 5 + 4 = 9
GPU2 (E4,E5): 3 + 4 = 7
GPU3 (E6,E7): 4 + 3 = 7
```

### 2.3.2 通信量估算

单条 token 向量大小：

```text
H * dtype_bytes = 4096 * 2 = 8192 bytes = 8 KB
```

本层 dispatch 的总数据量（全局）：

```text
T * K * H * dtype_bytes
= 16 * 2 * 4096 * 2
= 262,144 bytes
= 256 KB
```

同规模 gather 一次量级相近，因此 MoE 一层前向通信约：

```text
dispatch + gather ~= 512 KB（这个小例子）
```

> 真实训练中 T 往往是几千到几万，通信量会线性放大到 MB~GB 级。

---

## 2.4 Expert FFN 计算（每个 expert 独立）

对 expert `e`，输入是 `X_e: [T_e, H]`，输出 `Y_e: [T_e, H]`。

以 SwiGLU 为例：

```text
W_up[e]:   [H, F]
W_gate[e]: [H, F]
W_down[e]: [F, H]

U = X_e @ W_up[e]          -> [T_e, F]
G = X_e @ W_gate[e]        -> [T_e, F]
A = silu(G) * U            -> [T_e, F]
Y_e = A @ W_down[e]        -> [T_e, H]
```

每个 expert 的参数量：

```text
3 * H * F
= 3 * 4096 * 14336
= 176,160,768 (~176M params)
```

---

## 2.5 Gather All-to-All（第二次跨卡通信）

目标：把各 expert 输出送回 token 原始所属 GPU，并按路由权重合并。

对 token `ti`：

```text
output_i = sum_{j in TopK(i)} w_{i,j} * y_{i,j}
```

最终恢复形状：

```text
Y_flat: [T, H] = [16, 4096]
Y: [B, S, H] = [2, 8, 4096]
```

然后与残差连接，进入下一个子层/Block。

---

## 3. 反向传播数据流（训练最关键）

MoE 反向会镜像前向通信路径，通常包含：

1. `dY` 到各 expert 输出分支的加权拆分  
2. `A2A` 把梯度发回各 expert 所在 GPU  
3. 计算每个 expert 的参数梯度与输入梯度  
4. 再一次 `A2A` 把输入梯度送回 token 原始 GPU  
5. gate/router 相关梯度计算（含负载均衡 loss 分支）

可简化理解为：

```text
Forward:  token -> dispatch -> expert -> gather -> merge
Backward: dmerge -> dispatch_bwd -> dexpert -> gather_bwd -> dtoken
```

如果开了 DP/FSDP/ZeRO，还会叠加参数梯度同步（AllReduce / ReduceScatter）。

---

## 4. 端到端时间线（单层）

传统串行（未优化）：

```text
| Gate | A2A Dispatch | Expert GEMM | A2A Gather |
```

优化后常见形态（例如 chunk/tile overlap）：

```text
| Gate | (Dispatch chunk1 + Expert chunk0) 并发 ... | Gather |
```

核心目标：把通信隐藏在计算后面，降低 “GPU 等通信” 时间占比。

---

## 5. 工程实现中的关键中间张量

最常见中间数据结构：

```text
router_logits      [T, E]
topk_idx           [T, K]
topk_weight        [T, K]
token_per_expert   [E]         # 计数
permute_index      [T*K]       # dispatch 重排索引
send_counts        [P] or [P,P]
recv_counts        [P] or [P,P]
expert_input_buf   [sum_local_tokens, H]
expert_output_buf  [sum_local_tokens, H]
unpermute_index    [T*K]       # gather 回填索引
```

这些张量决定了通信开销和 kernel 可融合性，是系统优化重点。

---

## 6. 从小例子到真实规模的通信放大

小例子（本文件）：

```text
T=16, K=2, H=4096, bf16
单次 A2A 全局数据量 = 256 KB
```

真实训练常见量级（示例）：

```text
T=8192, K=2, H=4096, bf16
单次 A2A 全局数据量
= 8192 * 2 * 4096 * 2
= 134,217,728 bytes
= 128 MB

一次前向 MoE 层需 dispatch + gather ~= 256 MB
再加反向镜像通信，单层单步通信可到 512 MB 量级
```

这也是为什么 MoE 训练常出现“通信墙”。

---

## 7. 最小可复现伪代码（前向）

```python
def moe_forward(x):  # x: [T, H]
    logits = x @ W_gate                 # [T, E]
    probs = softmax(logits, dim=-1)     # [T, E]
    topk_w, topk_idx = topk(probs, k=K) # [T, K], [T, K]

    # 1) 构建 dispatch 索引与计数
    route_meta = build_route_meta(topk_idx, topk_w)

    # 2) dispatch: token -> expert 所在卡
    expert_in = all_to_all_dispatch(x, route_meta)

    # 3) 本地 expert 计算
    expert_out = run_local_experts(expert_in)

    # 4) gather: expert 输出 -> token 原卡
    token_out = all_to_all_gather(expert_out, route_meta)

    # 5) Top-K 合并
    y = weighted_merge(token_out, topk_w, topk_idx)
    return y
```

---

## 8. 常见坑位（实现时）

1. **负载不均衡**：`max(token_per_expert) / mean(...)` 过高导致长尾卡拖慢全局。  
2. **小包通信过多**：切分过细导致 NCCL 启动开销吞噬收益。  
3. **索引重排开销高**：permute/unpermute 内存访问不连续，L2 命中差。  
4. **容量溢出策略不一致**：drop token 或 reroute 会影响训练稳定性。  
5. **路由随机性与复现**：并行下 nondeterministic top-k 容易导致结果漂移。

---

## 9. 一句话总结

MoE 的本质是“用路由换稀疏激活”，而工程难点是“如何让 token 在专家之间高效流动”。  
因此，**计算过程**重点看 `Top-K + Expert FFN`，**数据流过程**重点看 `dispatch/gather + 重排索引 + 负载均衡`。

