---
title: "etcd"
date: 2026-09-23
tags: ["向量数据库", "Milvus","DevOps"]
---

# DevOps
**全称：Development & Operations**, 开发（Dev）+运维（Ops），不是一个工具，是一套**文化、流程、工程方法论**，目标：打通开发团队和运维团队的壁垒，让软件**更快、更稳定、更频繁地交付上线**。

## 理解
以前：开发写完代码丢给运维，开发只管功能，运维管服务器部署。双方经常扯皮：本地能跑，线上崩了。开发说运维环境问题，运维说代码问题。

DevOps：打破开发、运维的墙，**开发也要关心线上部署、监控；运维也要理解代码发布流程**，配合自动化工具，做到代码提交后自动测试、自动打包、自动部署、自动监控告警。

## 核心三大支柱
1. **文化**：团队协作，消除部门墙，共同对线上服务负责
2. **流程**：持续集成CI、持续交付CD（最核心）
    - CI（Continuous Integration，持续集成）：代码提交到仓库，自动拉代码、编译、单元测试，提前发现bug
    - CD（Continuous Delivery/Deployment，持续交付/持续部署）：CI通过后，自动打包，部署到测试环境；CD部署可自动上线生产
3. **工具链**（常用）
    - 代码仓库：Git / GitLab / GitHub
    - CI流水线：Jenkins、GitLab CI、GitHub Actions
    - 容器：Docker
    - 编排：Kubernetes(k8s)
    - 监控：Prometheus + Grafana
    - 日志：ELK

## 和Milvus关联
Milvus是云原生向量数据库，**Milvus本身就是基于DevOps理念设计**：
- 以容器方式部署在K8s；
- 组件（QueryNode、DataNode、RootCoord等）可弹性扩缩容；
- 有完善监控指标，方便自动化运维；
- 版本升级、滚动发布都依赖整套DevOps流水线。

## 补充区分（面试常问）
- Dev：开发，写业务代码
- Ops：运维，管机器、服务稳定性
- SRE（站点可靠性工程）：Google提出，**属于DevOps的一个分支**，侧重用工程手段保障线上稳定性，写自动化脚本、指标、故障预案。
