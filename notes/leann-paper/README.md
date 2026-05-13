# leann-paper — LEANN 低存储向量索引论文阅读

> **目标**：系统阅读和归档 LEANN: A Low-Storage Overhead Vector Index，关注低存储向量检索、RAG 本地部署、图索引压缩、embedding 按需重算与相关系统路线。
> **论文**：[arXiv:2506.08276](https://arxiv.org/abs/2506.08276)
> **代码**：[yichuan-w/LEANN](https://github.com/yichuan-w/LEANN)

## 状态

| 维度 | 状态 |
|---|---|
| 论文版本 | arXiv v2, 2025-11-25 |
| 阅读状态 | 已完成深度报告 |
| 重点结论 | LEANN 用图遍历下的 selective recomputation + 高度节点保留剪枝，把 76GB 文本库的索引存储压到 4GB，同时保持 HNSW 级别 RAG 准确率 |
| 后续价值 | 可作为本地 RAG / 冷数据检索 / 低存储 ANN 系统设计参考 |

## 进展时间线

| 日期 | 里程碑 | 关键结论 | 来源 note |
|---|---|---|---|
| 2026-05-13 12:51 | LEANN 论文深度阅读报告 | LEANN 在 60M passages 上相对 HNSW 存储 188GB→4GB；核心代价是检索延迟从 0.05s 到 2.48s，但端到端 RAG 仍由生成阶段主导 | [`2026-05-13_1251_leann_low_storage_vector_index_report`](./2026-05-13_1251_leann_low_storage_vector_index_report.md) |

## 下一步

| 优先级 | 方向 | 预期 |
|---|---|---|
| P0 | 复现实验最小闭环 | 在小规模本地语料上验证 LEANN vs FAISS/HNSW 的存储、Recall@3、检索延迟 |
| P1 | 评估 GPU/CPU embedding 重算瓶颈 | 拆分 PQ lookup、文本读取/tokenize、embedding recompute 三段耗时 |
| P1 | 研究 AMD/本地硬件部署 | 若要迁移到 AMD GPU，重点看 embedding 模型推理吞吐与动态 batch 策略 |

## 文件索引

| 主题 | 文件 |
|---|---|
| LEANN 论文深度中文报告 | [`2026-05-13_1251_leann_low_storage_vector_index_report.md`](./2026-05-13_1251_leann_low_storage_vector_index_report.md) |
