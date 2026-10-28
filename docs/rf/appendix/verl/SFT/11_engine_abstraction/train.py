# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置 + 日志追踪 + 梯度检查点 + remove_padding + 动态 batch + 手算 loss + 分布式进程组
  + 数据并行(DDP) + FSDP 模型分片 + 引擎抽象层(BaseEngine/TrainingWorker/EngineRegistry) 版 SFT 训练
—— verl/learning/sft/code/11_engine_abstraction/train.py

==================== 【新增·模块1~9】(保持不变, 见之前模块顶部说明) ====================
模块1 配置系统 / 模块2 Tracking / 模块3 梯度检查点 / 模块4 remove_padding / 模块5 动态 batch /
模块6 手算 loss / 模块7 分布式进程组 / 模块8 数据并行(DDP+no_sync+*dp_size) / 模块9 FSDP 分片。

==================== 【新增·模块10】引擎抽象层: train.py 瘦身成三层中的最外层("fit"层) ====================
在模块9(10_fsdp)基础上继续改动(**纯结构重构, 数值语义不变, loss 轨迹应与模块9 逐位一致**):
  - 模块9 的 main() 里还有 AutoModel 加载、梯度检查点、FSDP 包裹、DataLoader、optimizer、lr_scheduler、
    训练 for 循环、clip_grad_norm_、all_reduce、Tracking 构造与 log、save_pretrained —— 全堆在 main 一锅烩。
  - 本模块把这些全下沉到 worker + engine, main() 变成 ~25 行, 结构对应 verl 的 **fit / train_batch / engine 三层**:
      (1) train.py.main  = verl SFTTrainer.fit 的最外层调度(只 init_pg + set_device + worker.fit());
      (2) worker.py      = verl TrainingWorker(建 engine、注入 loss、外循环、_postprocess、存盘);
      (3) engine.py      = verl BaseEngine(train_batch 骨架) + FSDPEngine/AutoModelEngine + EngineRegistry。
  - 删掉的(已下沉): AutoModelForCausalLM 加载、gradient_checkpointing、FSDP 包裹、DataLoader、optimizer、
    lr_scheduler、训练 for 循环、clip_grad_norm_、optimizer.step、Tracking 构造与 log、save_pretrained ——
    全部由 worker.fit() / engine.initialize() / engine.train_batch() / worker._postprocess() / worker._save_model() 承接。

意义: 把模块9 "压平的 v0 主循环" 重新拆成 verl 的三层 —— 这就是 verl 能在 fsdp/megatron/automodel 不同后端间
  切换、又能把 loss 公式与训练机制解耦的架构根基。后续学 megatron/torch.compile 只新增一个 Engine 子类, worker 与主循环不动。
对应 verl: verl/trainer/sft_trainer.py 的 SFTTrainer.fit(最外层) + verl/workers/engine_workers.py 的 TrainingWorker
  + verl/workers/engine/base.py 的 BaseEngine + EngineRegistry。
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config, config_to_yaml_str   # 【模块1】配置系统
from dist import initialize_global_process_group, destroy_global_process_group, is_rank0   # 【模块7】进程组
from worker import TrainingWorker   # 【新增·模块10】引入 worker(worker 持 engine, engine 持 FSDP model)


# 【模块1 沿用】解析命令行: --config <yaml> + 其余 key.path=value 覆盖
def parse_cli():
    args = sys.argv[1:]
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    overrides = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--config":
            yaml_path = args[i + 1]
            i += 2
        elif "=" in a and not a.startswith("--"):
            overrides.append(a)
            i += 1
        else:
            i += 1  # 忽略未知参数
    return load_config(yaml_path, overrides), overrides


def main():
    # 【模块1】配置来自 yaml + 覆盖, 不再是 argparse
    cfg, overrides = parse_cli()

    # 【模块7 沿用】分布式进程组初始化 —— torchrun 启动时建 NCCL 组, 单进程回退。
    #   对应 verl run_sft(): initialize_global_process_group(); trainer=SFTTrainer(config); trainer.fit()
    local_rank, rank, world_size = initialize_global_process_group(
        backend=cfg.trainer.backend, timeout_second=36000,
    )
    cfg.trainer.world_size = world_size   # 用真实 world_size 覆盖 config 占位值(verl 也是运行时从 env 读)
    device = f"cuda:{local_rank}"
    # 【模块7 沿用】模型 init 用同一个 seed(各 rank 一致) -> DDP/FSDP 梯度同步才有意义。
    #   注意是 cfg.trainer.seed(同 seed), 不是 +local_rank(那会让各 rank 初始化不同, FSDP sync_module_states 虽能纠正但 DDP/automodel 路径会发散)。
    torch.manual_seed(cfg.trainer.seed)
    os.makedirs(cfg.trainer.output_dir, exist_ok=True)

    # 【模块7 沿用】rank0 判定: 只有 rank0 打印配置(对应 verl sft_trainer.py L77 `if self.rank == 0: print(self.config)`)
    is_main = is_rank0()
    if is_main:
        print("=" * 60)
        print("最终生效配置(配置即实验记录):")
        if overrides:
            print("  命令行覆盖:", overrides)
        print(config_to_yaml_str(cfg))
        print("=" * 60)
        print(f"[dist] world_size={world_size}, rank={rank}, local_rank={local_rank}, "
              f"device={device}, backend={cfg.trainer.backend}")

    # 【新增·模块10】三层架构: worker 持 engine, engine 持 FSDP model —— main 只调 fit()。
    #   worker.__init__ 里: 建 dataloader -> EngineRegistry.new(cfg.trainer.engine_backend) 选后端建引擎 ->
    #   engine.initialize()(加载模型+FSDP+优化器) -> engine.build_scheduler()。
    #   worker.fit() 里: 外循环调 engine.train_batch(骨架: zero_grad->forward_backward_batch->optimizer_step)。
    worker = TrainingWorker(cfg, local_rank, world_size)
    worker.fit()

    # 【模块7 沿用】销毁进程组(所有 rank 都调, 对应 verl run_sft 末尾 destroy_global_process_group)。
    #   不调的话 NCCL communicator 不释放, torchrun 进程挂住不退出。
    destroy_global_process_group()


if __name__ == "__main__":
    main()
