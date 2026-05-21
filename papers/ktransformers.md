# KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models

> **发表:** SOSP '25  
> **机构:** 清华大学 / KVCache.AI  
> **arXiv:** 已开源 → https://github.com/kvcache-ai/ktransformers  
> **核心贡献:** CPU+GPU 异构推理，单台消费级机器运行 DeepSeek-V3/R1 等超大 MoE 模型

---

## 1. 核心问题：超大 MoE 模型的民主化推理

### 1.1 现实场景：普通开发者如何运行 DeepSeek-V3？

```
DeepSeek-V3（671B 参数）的推理需求：

完整加载到 GPU（BF16）：
  671B × 2 bytes = 1.34 TB VRAM
  需要：~17 张 H100 80GB  → 成本 > $50万
  
量化到 INT4 / INT8：
  671B × 0.5 bytes = 335GB VRAM
  需要：~5 张 H100 80GB  → 仍然极贵

问题：DeepSeek-V3 性能媲美 GPT-4，但普通开发者用不起

KTransformers 目标：
  用 1~2 张消费级 GPU（24~48 GB）+ 大量 CPU 内存（256 GB+）
  实现可用的推理性能（> 5 tokens/sec）
```

### 1.2 MoE 架构的特殊性使异构推理成为可能

```
Dense 模型（LLaMA-70B）的问题：
  每次 forward：必须访问所有 70B 参数
  → 无法部分卸载（每个矩阵都会被访问）

MoE 模型（DeepSeek-V3）的机会：
  256 个 Expert，每 token 只激活 8 个
  → 每次 forward：只访问 8/256 = 3% 的 Expert 参数！
  
  可以把 Expert 放在 CPU 内存，只按需加载到 GPU
  → 每次推理只需要 ~3% 的 Expert 参数在 GPU 上
  
  Attention 参数：~40GB（必须在 GPU）
  Expert 参数：~1.3TB（可以在 CPU 内存！）
```

---

## 2. KTransformers 系统设计

### 2.1 异构内存层次

```
KTransformers 的内存分层：

Level 1：GPU VRAM（24~48 GB）
  ├─ 所有 Attention 参数（Q/K/V/O Projection）
  ├─ KV Cache（动态分配）
  └─ 当前激活的 8 个 Expert 参数（热缓存）

Level 2：CPU DRAM（256~512 GB DDR5）
  ├─ 所有 Expert 参数（~1.3TB × INT4量化 = ~335GB）
  └─ Expert 权重缓存（LRU）

Level 3：NVMe SSD（2TB+）
  └─ 模型参数冷存储（offloading 的最终后备）
  
数据流：
  SSD → CPU DRAM → GPU VRAM → 计算
  （按需、提前预加载）
```

### 2.2 核心技术：Expert 预取与缓存

```python
class KTransformersExpertManager:
    def __init__(self, model_path, gpu_memory_gb=48, cpu_memory_gb=256):
        # GPU 上只放 Attention + 热 Expert
        self.gpu_cache = ExpertLRUCache(
            capacity_mb=gpu_memory_gb * 1024 * 0.4  # 40% GPU 给 Expert
        )
        
        # CPU 放全量 Expert（INT4 量化后约 335 GB）
        self.cpu_cache = ExpertLRUCache(
            capacity_mb=cpu_memory_gb * 1024 * 0.8  # 80% CPU 给 Expert
        )
        
        self.prefetch_predictor = ExpertAccessPredictor()
    
    def get_expert(self, expert_id, layer_id):
        """多级缓存查找"""
        # Level 1: GPU VRAM
        if expert_id in self.gpu_cache:
            return self.gpu_cache[expert_id]  # 纳秒级
        
        # Level 2: CPU DRAM（需要 PCIe 传输）
        if expert_id in self.cpu_cache:
            # PCIe 传输：~5ms 对于 4MB Expert
            self.async_load_to_gpu(expert_id)
            return self.cpu_cache[expert_id]  # CPU 上计算作为备选
        
        # Level 3: SSD（慢，应尽量避免）
        expert = self.load_from_disk(expert_id)
        self.cpu_cache[expert_id] = expert
        return expert
    
    def prefetch_experts(self, current_layer, routing_predictions):
        """基于预测的提前预取"""
        for future_layer in range(current_layer + 1, current_layer + 3):
            predicted_experts = self.prefetch_predictor.predict(
                layer_id=future_layer,
                current_routing=routing_predictions
            )
            for eid in predicted_experts:
                if eid not in self.gpu_cache:
                    self.async_load_to_gpu(eid)  # 提前加载
```

### 2.3 计算路径优化

```
推理计算流：

Input Token
    ↓
[Attention Layer] ← 在 GPU 上执行（VRAM 中参数）
    ↓
[Router] ← 决定 8 个 Expert
    ↓
[Expert Dispatch]
    ├─ 命中 GPU Cache：直接在 GPU 上执行 Expert ← 快（~0.5ms）
    └─ 命中 CPU Cache：在 CPU 上执行 Expert ← 慢（~15ms）
    
    ↓
[合并 Expert 输出]
    ↓
[下一层 Attention]

关键优化：
  1. 最大化 GPU Cache 命中率 → 减少 CPU 执行的 Expert 数
  2. CPU Expert 计算与 GPU Attention 计算并行
  3. 使用 AVX512 指令加速 CPU 上的 Expert GEMM
```

### 2.4 量化策略

```
KTransformers 量化配置（针对 DeepSeek-V3）：

Attention 参数（GPU）：BF16 → 保持精度
Expert 参数（CPU）：INT4（GGUF 格式）→ 压缩 4x

量化选择策略：
  ┌─────────────────────────────────────────┐
  │ 敏感层（Attention）：BF16/FP16          │
  │ Expert（大量参数）：INT4，可接受精度损失 │
  │ LayerNorm/Embedding：FP32              │
  └─────────────────────────────────────────┘

精度对比（DeepSeek-V3, MMLU benchmark）：
  BF16 原始：89.5%
  KTransformers INT4：88.2%（-1.3%，可接受）
```

---

## 3. 性能实验结果

### 3.1 硬件配置与性能

| 硬件配置 | 模型 | 推理速度 | 成本 |
|---------|------|---------|------|
| RTX 4090 × 2 + 512GB DDR5 | DeepSeek-V3 | **15~20 tokens/s** | ~$8K |
| RTX 4090 × 1 + 256GB DDR5 | DeepSeek-R1 | **8~12 tokens/s** | ~$5K |
| A100 × 4 + 512GB DDR5 | DeepSeek-V3 | **35~50 tokens/s** | ~$20K |
| H100 × 8（完整方案） | DeepSeek-V3 | **100+ tokens/s** | ~$100K+ |

> KTransformers 的价值：**$5~8K 的机器跑起 DeepSeek-V3**，速度够用（> 5 tok/s）

### 3.2 缓存命中率分析

```
不同请求类型的 Expert 命中率：

聊天场景（重复性高）：
  GPU Cache 命中率：~65%
  CPU Cache 命中率：~30%
  SSD 访问：~5%
  
  有效速度：~15 tokens/s

随机代码生成（低重复性）：
  GPU Cache 命中率：~40%
  CPU Cache 命中率：~45%
  SSD 访问：~15%
  
  有效速度：~8 tokens/s

关键发现：MoE 模型的 Expert 访问模式有很高的局部性！
  → 相同类型的输入倾向于激活相同的 Expert
  → LRU 缓存策略在实践中非常有效
```

### 3.3 与其他方案对比

| 方案 | 硬件 | DeepSeek-V3 速度 | 可用性 |
|------|------|---------------|------|
| **KTransformers** | 4090×2 + 512GB | **15-20 tok/s** | ✅ 个人可负担 |
| llama.cpp | 4090×2 + 512GB | 5-8 tok/s | ✅ 但更慢 |
| vLLM 完整 | H100×8 | 100+ tok/s | ❌ 成本极高 |
| Ollama (CPU 纯) | CPU 仅 | 0.5-2 tok/s | ✅ 但太慢 |

---

## 4. 核心创新总结

### 4.1 Expert 局部性的发现与利用

```
论文的关键洞察（SOSP 值得关注的发现）：

观察 1：MoE 模型的 Expert 访问遵循幂律分布
  ├─ 少量 Expert 被频繁访问（头部 ~20% Expert 占 ~60% 访问）
  └─ 大量 Expert 极少被访问（尾部）
  
观察 2：相同类型输入倾向于激活相同 Expert
  ├─ 代码问题 → 固定的编程相关 Expert
  ├─ 数学问题 → 数学推理相关 Expert
  └─ 日常对话 → 通用 Expert
  
利用方式：
  ├─ LRU 缓存（GPU + CPU 双级）
  ├─ 内容感知预取（基于输入类型预测 Expert 集合）
  └─ 静态热点 Expert 永驻 GPU
```

### 4.2 CPU GEMM 优化

```python
# CPU 上的 Expert 计算优化（AVX512 加速）

class OptimizedCPUExpert:
    def forward(self, x, expert_weight_int4):
        """
        使用 AVX512 + INT4 矩阵乘法
        """
        # 反量化（INT4 → BF16，在 CPU 上）
        weight_bf16 = dequantize_int4_to_bf16(expert_weight_int4)
        
        # AVX512 优化的矩阵乘法
        output = avx512_gemm(x.to('cpu'), weight_bf16)
        
        # 异步传回 GPU（下一层需要时）
        return output.to('cuda', non_blocking=True)
```

---

## 5. 对 AI Infra 工程师的启示

### 5.1 设计原则

1. **利用 MoE 的稀疏性做分层缓存**
   - Expert 激活稀疏 → 可以用 CPU 内存做第二级缓存
   - 这是 Dense 模型无法做到的

2. **Expert 访问局部性是缓存的理论基础**
   - 用简单的 LRU 就能获得很高命中率
   - 内容感知预取可以进一步提升

3. **CPU-GPU 协同计算不是性能瓶颈，是性能基础**
   - CPU Expert + GPU Attention 并行执行
   - PCIe 传输可以被计算覆盖

### 5.2 工程实现要点

```python
# KTransformers 集成到 HuggingFace Transformers

from ktransformers import KTransformersModel

model = KTransformersModel.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    
    # 硬件配置
    cpu_memory_gb=256,
    gpu_memory_gb=48,
    
    # 量化配置（Expert 用 INT4，Attention 用 BF16）
    quantization_config={
        'experts': 'int4',
        'attention': 'bf16',
        'layernorm': 'fp32',
    },
    
    # 缓存策略
    cache_policy='lru',
    prefetch_enabled=True,
    prefetch_window=2,  # 提前 2 层预取
)

# 使用方式与标准 HuggingFace 完全一致
output = model.generate(input_ids, max_length=512)
```

---

## 6. 横向对比总结

| 论文 | 适用场景 | 硬件要求 | 推理速度 | 核心技术 |
|------|---------|---------|---------|---------|
| **KTransformers** | 个人/小团队 | 消费级 GPU + 大内存 | **15-20 tok/s** | CPU+GPU 异构缓存 |
| **MegaScale-Infer** | 云服务 | 万卡 GPU | **100+ tok/s** | 分离式 EP |
| **vLLM** | 中等规模 | 多张 A100/H100 | 50-100 tok/s | PagedAttention |
| **llama.cpp** | 纯 CPU/边缘 | 仅 CPU | 0.5-5 tok/s | GGUF 量化 |

**KTransformers 的独特价值：填补了「没有 H100 但想用顶级模型」的空白**

---

## 延伸阅读

- 🔧 **KTransformers GitHub** → https://github.com/kvcache-ai/ktransformers
- 📄 **llama.cpp** - 纯 CPU 推理的先驱 → https://github.com/ggerganov/llama.cpp
- 📄 **PowerInfer** - 类似 CPU+GPU 混合思路 → https://arxiv.org/abs/2312.12456
- 📄 **MegaScale-Infer** - 云端版本 → SIGCOMM'25

---

*笔记整理于 2026-03-07，基于 SOSP'25 论文及开源代码*
