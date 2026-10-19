---
title: "配置参数"
date: 2026-10-19
categories:
  - config
tags:
  - config
---

## 核心成分

1. 一个config.yaml文件，用于放置配置参数。
   1. data相关参数，用于配置与数据有关的信息。
   2. model相关参数，用于配置与模型相关的信息。
   3. trainer参数，用于存储与训练相关的信息。
   4. optim参数，用于配置与优化器相关的信息。
2. 一个config.py文件，用于管理配置信息
   1. 设置各个表格的schema。使用一个SFTConfig，嵌套DataConfig,ModelCofig,OptimConfig,TrainConfig。
   2. 设计load_config读取config.yaml，使用递归将dict填进dataclass。
   3. 设计_apply_override，支持命令行使用“key.path=value”的形式覆盖config.yaml的配置。