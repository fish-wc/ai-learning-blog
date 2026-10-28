# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
分布式进程组初始化 —— verl/learning/sft/code/8_distributed_init/dist.py

==================== 【新增·模块7】分布式进程组 (torch.distributed) ====================
在模块6(单进程)基础上: 读 torchrun 注入的 RANK/WORLD_SIZE/LOCAL_RANK, 建 NCCL 进程组,
按 local_rank 选卡, 返回 (local_rank, rank, world_size)。保留单进程回退(方便烟测)。

对应 verl: verl/utils/distributed.py 的 initialize_global_process_group / destroy_global_process_group。
verl run_sft() 的三段式: initialize_global_process_group() -> trainer.fit() -> destroy_global_process_group()。

【和 verl 的区别(简化点)】
  - verl backend = f'cpu:gloo,{device}:{nccl_backend}'(CPU+GPU 双组, 给 vLLM 权重同步等用),
    我们只用单一 nccl backend —— 学 SFT 训练循环不需要 CPU 组。
  - verl 用 Ray + DIST_INIT_METHOD 注入 init_method, 我们用 torchrun 注入的
    MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK(init_process_group 自动读 env)。
    保留了"进程组从环境变量读 rank/world_size"这个核心结构。
  - 单进程回退(RANK 不在 env 时返回 (0,0,1)): verl 永远在 Ray/torchrun 下跑没有这个;
    这里加是为了 `python train.py` 烟测仍能工作, 不算 verl 特性。
"""
import os
# 【环境坑·必设·模块7】CUDA_DEVICE_ORDER=PCI_BUS_ID: 让 CUDA 设备编号按 PCI 总线序(= nvidia-smi 序)排。
#   默认 FASTEST_FIRST 会把两张 3090 排前、2080Ti 排后 -> CUDA_VISIBLE_DEVICES=0,2 取到的是 {3090, 2080Ti}
#   而非 {3090, 3090}, rank1 落到 11GB 的 2080Ti 上必 OOM。必须在首个 torch.cuda 调用前设(用 setdefault,
#   不覆盖用户显式设置)。verl 侧由 Ray/launcher 环境保证; 本学习版在 dist.py 里兜底。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from datetime import timedelta

import torch
import torch.distributed as dist


def initialize_global_process_group(backend: str = "nccl", timeout_second: int = 36000):
    """初始化全局进程组。对应 verl initialize_global_process_group(timeout_second=36000)。

    torchrun 启动时已把 MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK 注入 os.environ,
    init_process_group 会自动读这些 env, 无需显式传 rank/world_size/init_method。
    返回 (local_rank, rank, world_size) —— 和 verl 一致(verl 也是返回这三个)。

    单进程回退: 没被 torchrun 启动(RANK 不在 env, 如 `python train.py` 烟测)时不建组, 返回 (0,0,1)。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_second),
            )
        # 【关键】set_device(local_rank): NCCL communicator + 后续 model.to(device)/batch.to(device)
        #   都要落在本 rank 的卡上。不 set_device 会落到默认 cuda:0, 跨卡训练显存错位。
        #   对应 verl get_torch_device().set_device(local_rank)。
        torch.cuda.set_device(local_rank)
    else:
        # 单进程回退: 不建组
        rank, world_size, local_rank = 0, 1, 0
    return local_rank, rank, world_size


def destroy_global_process_group():
    """对应 verl destroy_global_process_group。所有 rank 都要调, 否则 NCCL 进程挂住、torchrun 不退出。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0() -> bool:
    """rank 0 判定(单进程时也算 rank 0)。对应 verl 里到处用的 `if self.rank == 0`。"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    """当前 world_size(单进程时 1)。对应 verl engine.get_data_parallel_size() 的最简版。"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """当前 rank(单进程时 0)。对应 verl engine.get_data_parallel_rank() 的最简版。"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0
