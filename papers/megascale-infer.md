# MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

> **发表:** SIGCOMM '25  
> **机构:** ByteDance  
> **场景:** 大规模 MoE 推理服务系统  
> **核心贡献:** 分离式专家并行（Disaggregated EP）实现万卡规模 MoE 高效推理

---

## 1. 核心问题：MoE 推理服务的资源利用困境

### 1.1 MoE 推理的结构性矛盾

```
传统 LLM 推理服务（Dense 模型）：
  所有 token → 相同路径 → 相同计算量
  → 资源使用可预测，负载均衡简单

MoE 推理服务的特殊性：
  不同 token → 不同 Expert → 计算量差异大
  
  场景 1：短输入，长输出（聊天场景）
    Prefill: 需要大量 KV Cache，Expert 计算少
    Decode:  每次 1 token，Expert 利用率极低（稀疏！）
  
  场景 2：批处理场景（搜索、推荐）
    大 batch，Expert 被频繁访问
    Expert 并行度高，通信量也高
  
  核心矛盾：
    Prefill 阶段 → 需要高带宽内存（KV Cache），Expert 相对空闲
    Decode 阶段  → Expert 成为瓶颈，但 KV Cache 使用低
    → 传统统一部署方式，资源利用率极低（< 30%）
```

### 1.2 通信开销随规模线性增长

```
Expert Parallel 通信量分析：

单个请求的 Expert Dispatch：
  通信量 = batch_tokens × d_model × 2 bytes
  
并发请求数 N，EP 度 = K：
  总通信量 = N × seq_len × d_model × 2 × 2（dispatch + gather）
  
  EP=8：   通信量 ≈ 1x
  EP=32：  通信量 ≈ 2-3x（跨节点，带宽降低）
  EP=128： 通信量 ≈ 8-12x（跨 Pod，严重拥塞）

问题：扩大 EP 度 → 通信成为主导瓶颈
     但不扩大 EP 度 → Expert 无法容纳在更少 GPU 上
     → 死锁困境！
```

---

## 2. MegaScale-Infer 核心设计：分离式专家并行

### 2.1 核心思想：Prefill / Decode 解耦 + Expert 分离

```
传统 Coupled 部署：

         同一批 GPU
    ┌─────────────────────────────┐
    │  Attention  +  Expert FFN   │ Prefill/Decode 混用同一组 GPU
    │  KV Cache  +  Expert Params │ 资源争抢严重
    └─────────────────────────────┘

MegaScale-Infer 分离部署：

Prefill Cluster：                 Decode Cluster：
┌──────────────────┐              ┌──────────────────────────────┐
│  Attention GPU   │              │  Expert GPU (Large Pool)     │
│  KV Cache 重     │    EP        │  Expert Params 存储         │
│  Expert 轻度用   │◄────────────►│  高并发，Expert 密集访问     │
└──────────────────┘    A2A       └──────────────────────────────┘
         ↑                                      ↑
    Prefill 阶段                          Decode 阶段
（高带宽内存需求）                     （高 Expert 计算需求）

关键：两个集群共享 Expert 参数，但 KV Cache 不共享
```

### 2.2 分离式 Expert Parallel（Disaggregated EP）

```python
# MegaScale-Infer 的分离式 EP 设计（概念）

class DisaggregatedExpertParallel:
    def __init__(self, prefill_gpus, decode_gpus, expert_gpus):
        # 三类角色分离
        self.prefill_nodes = prefill_gpus    # 负责 Prefill 的 Attention
        self.decode_nodes  = decode_gpus     # 负责 Decode 的 Attention  
        self.expert_nodes  = expert_gpus     # 专门存储和计算 Expert
    
    def prefill(self, input_tokens):
        """Prefill 阶段：计算密集，Expert 访问少"""
        # Attention 在 prefill_nodes 本地计算
        attn_out = self.local_attention(input_tokens)
        
        # Expert 调用：发送到 expert_nodes
        expert_out = self.remote_expert_call(
            attn_out,
            destination=self.expert_nodes,
            mode='batch'  # 批量调用，效率高
        )
        
        return expert_out
    
    def decode(self, kv_cache, prev_tokens):
        """Decode 阶段：内存密集，频繁访问 Expert"""
        # KV Cache 在 decode_nodes 本地
        attn_out = self.local_attention_with_cache(kv_cache, prev_tokens)
        
        # Expert 调用：异步发起，减少等待
        expert_future = self.async_expert_call(
            attn_out,
            destination=self.expert_nodes,
            mode='streaming'  # 流式调用，减少延迟
        )
        
        return expert_future.result()
```

### 2.3 Expert 节点的内部架构

```
Expert 节点（专门的 Expert 计算集群）：

┌────────────────────────────────────────────────────────┐
│                  Expert Node Pool                       │
├────────────────────────────────────────────────────────┤
│  Expert 0,1,...,7    ← GPU 0   (NVLink 内部连接)       │
│  Expert 8,9,...,15   ← GPU 1                           │
│  ...                                                    │
│  Expert 248,...,255  ← GPU 31                          │
├────────────────────────────────────────────────────────┤
│  Request Router                                         │
│  ├─ 来自 Prefill 集群的 token → 批处理 → 高效 GEMM      │
│  └─ 来自 Decode 集群的 token  → 流式  → 低延迟响应      │
└────────────────────────────────────────────────────────┘

优势：
  Expert 节点专注于 GEMM 计算，GPU 利用率极高（> 80%）
  不受 KV Cache 内存竞争的影响
  可以独立扩缩容（只增加 Expert GPU）
```

### 2.4 网络通信优化

```
Prefill/Decode 集群与 Expert 集群间的通信：

传统方式：All-to-All（需要所有节点参与）
MegaScale-Infer 方式：P2P（点对点，按需）

通信模式对比：
  All-to-All N nodes:    N × (N-1) 对通信
  P2P (M prefill/decode → K experts): M × K 对通信
                                      → 通信量可控

负载感知路由：
  请求 → Router → 找负载最低的 Expert GPU
          ↓
    采用加权轮询 (WRR) 算法
    权重 = 1 / (current_queue_len + expected_compute_time)
```

---

## 3. 调度策略

### 3.1 请求调度的优化目标

```
SLO（Service Level Objectives）约束：
  TTFT (Time to First Token)  ≤ 1000ms   → Prefill 时间限制
  TBT  (Time Between Tokens)  ≤ 100ms    → Decode 速度
  端到端 P99 延迟              ≤ 5s       → 整体体验

挑战：
  长 Prefill 请求占用 Expert 节点 → 影响 Decode 的 TBT
  短 Decode 请求大量并发 → 积压 Expert 队列
  
MegaScale-Infer 的优先级调度：
  Decode 请求 > Prefill 请求（因为 Decode 对 TBT 更敏感）
  长 Prefill 请求分片处理（Chunked Prefill）
```

### 3.2 Chunked Prefill 与 Expert 交互

```python
class ChunkedPrefillScheduler:
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
    
    def schedule_prefill(self, request):
        """将长 Prefill 分成多个 chunk，与 Decode 交错执行"""
        chunks = request.input_tokens.split(self.chunk_size)
        
        for chunk in chunks:
            # 1. 执行当前 chunk 的 Attention
            attn_out = self.attention(chunk)
            
            # 2. 访问 Expert（异步，不阻塞 Decode）
            expert_future = self.async_expert_dispatch(attn_out)
            
            # 3. 让 Decode 请求穿插执行
            yield_to_decode_if_needed()
            
            # 4. 收集 Expert 结果
            expert_out = expert_future.result()
        
        return combine_chunks(results)
```

---

## 4. 性能实验结果

### 4.1 核心指标

| 指标 | MegaScale-Infer | 传统部署 | 改进 |
|------|----------------|---------|------|
| **GPU 利用率** | **78%** | 32% | **+146%** |
| **吞吐量** | **3.2x** | 基准 | **3.2x ↑** |
| **TTFT P50** | 350ms | 580ms | **40% ↓** |
| **TTFT P99** | 1100ms | 2200ms | **50% ↓** |
| **成本/QPS** | **0.45x** | 1.0x | **55% ↓** |

### 4.2 扩展性测试

```
在 10,000+ GPU 规模下：

传统 EP（All-to-All）：
  GPU 翻倍 → 吞吐量 +80%（不完美线性）
  延迟也在增加（通信拥塞）

MegaScale-Infer（分离式 EP）：
  GPU 翻倍 → 吞吐量 +95%（接近线性！）
  延迟保持稳定（P2P 通信不受全局影响）
```

### 4.3 不同场景下的效率

```
场景 1：聊天应用（长对话，短输出）
  Expert 复用率高 → 分离式架构优势明显
  成本节省：45~55%

场景 2：代码生成（长输出，重 Decode）
  Decode 阶段 Expert 访问频繁
  分离式架构：Expert 节点专注 GEMM → GPU 利用率 80%+

场景 3：批量推断（高吞吐，低延迟要求）
  大 batch → Expert 节点的 GEMM 效率最高
  与传统方案相当，但可弹性扩缩容
```

---

## 5. 与其他推理系统的对比

### 5.1 架构对比

| 系统 | 架构模式 | Expert 部署 | KV Cache | 扩展性 |
|------|---------|-----------|---------|-------|
| **MegaScale-Infer** | **分离式** | 独立 Expert 节点 | 在 Decode 节点 | **接近线性** |
| **vLLM** | 统一部署 | 与 Attention 同 GPU | 同 GPU | 受 GPU 内存限制 |
| **DeepSpeed-MII** | 统一部署 | 固定 EP 分组 | 同 GPU | 中等 |
| **TensorRT-LLM** | 统一部署 | TP 分割 | 同 GPU | 受网络带宽限制 |

### 5.2 与 Janus（推理分离另一篇论文）的对比

```
Janus [arxiv'25]: Disaggregating Attention and Experts for Scalable MoE Inference
  → 同样的分离式思想，但 Janus 是学术论文（小规模）

MegaScale-Infer：
  → 工业实现，万卡规模验证
  → 更完整的调度策略（Chunked Prefill + WRR + SLO 保证）
  → 容错机制更完善

两者互补：Janus 提供理论框架，MegaScale-Infer 提供工程实践
```

---

## 6. 对 AI Infra 工程师的启示

### 6.1 MoE 推理系统设计原则

1. **Prefill 和 Decode 应该解耦**
   - 两者的资源需求模式截然不同
   - 统一部署导致严重的资源浪费

2. **Expert 应该成为独立的计算资源**
   - Expert 计算 = GEMM → GPU 计算密集型
   - 与 KV Cache 内存需求解耦，独立扩缩容

3. **通信模式的选择决定扩展性**
   - All-to-All：简单但不可扩展
   - P2P：复杂但可扩展到万卡

### 6.2 集成到现有推理框架

```python
# vLLM 插件化集成思路（概念）

class DisaggregatedMoEScheduler:
    def __init__(self, vllm_engine, expert_pool_config):
        self.engine = vllm_engine
        self.expert_pool = RemoteExpertPool(expert_pool_config)
    
    def schedule_request(self, request):
        # 根据请求类型分配资源
        if request.is_prefill():
            return self.schedule_prefill(request)
        else:
            return self.schedule_decode(request)
    
    def schedule_prefill(self, request):
        # Chunked Prefill + 远程 Expert 调用
        for chunk in request.chunks(size=512):
            attn = self.local_attention(chunk)
            expert_out = self.expert_pool.call_async(attn)
            ...
    
    def schedule_decode(self, request):
        # 优先保证 TBT
        attn = self.local_attention_with_cache(request)
        expert_out = self.expert_pool.call_priority(attn)  # 高优先级
        ...
```

---

## 7. 关键技术清单（阅读 PDF 时重点关注）

| 章节 | 核心内容 | 阅读价值 |
|------|---------|---------|
| **§3 Architecture** | 分离式 EP 的具体实现 | ⭐⭐⭐⭐⭐ |
| **§4 Communication** | P2P vs All-to-All 的性能模型 | ⭐⭐⭐⭐⭐ |
| **§5 Scheduling** | SLO 保证下的调度算法 | ⭐⭐⭐⭐⭐ |
| **§6 Evaluation** | 万卡实验数据 | ⭐⭐⭐⭐⭐ |
| **§7 Deployment** | 字节跳动生产实践 | ⭐⭐⭐⭐⭐ |

---

## 延伸阅读

- 📄 **Janus** - 学术版分离式推理 → [arxiv'25] Janus: Disaggregating Attention and Experts
- 📄 **MegaScale-MoE** - 同团队训练系统 → EuroSys'26
- 📄 **PD 解耦（DistServe）** - Prefill/Decode 解耦先驱 → https://arxiv.org/abs/2401.09670
- 🔧 **vLLM** - 集成目标框架 → https://github.com/vllm-project/vllm

---

*笔记整理于 2026-03-07，基于 SIGCOMM'25 论文信息*
