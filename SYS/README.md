# SYS 目录说明

## 目录概述

本目录包含 Prompt Composer 项目的系统架构文档和优化总结，用于记录项目的核心设计理念、技术架构和性能优化方案。

## 文件结构

```
SYS/
├── SYSTEM_ARCHITECTURE.md      # 系统架构设计文档
├── ARCHITECTURE_OPTIMIZATION_SUMMARY.md  # 架构优化总结文档
└── README.md                   # 本文件
```

## 文件说明

### 1. SYSTEM_ARCHITECTURE.md

系统架构设计文档，包含：
- 项目整体架构设计
- 核心模块划分
- 数据流和交互流程
- 关键技术选型
- 模块间依赖关系

### 2. ARCHITECTURE_OPTIMIZATION_SUMMARY.md

架构优化总结文档，包含：
- 性能优化方案
- 压缩算法优化策略
- 模型选择与调优
- 资源利用优化
- 并行处理方案

## 使用建议

- 新开发者请先阅读 `SYSTEM_ARCHITECTURE.md` 了解项目架构
- 关注性能优化的开发者请参考 `ARCHITECTURE_OPTIMIZATION_SUMMARY.md`
- 文档会随项目迭代持续更新

## 相关目录

- `core/` - 核心模块实现
- `model/` - 预训练模型存储
- `eval_results/` - 评估结果输出
- `tests/` - 测试脚本目录

---

*Prompt Composer - 智能提示词压缩与优化框架*
