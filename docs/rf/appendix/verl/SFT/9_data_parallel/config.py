# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置系统 —— verl/learning/sft/code/2_config_system/config.py

==================== 【新增·模块1】结构化配置系统 ====================
【和基础版(1_base_sft)的区别】
基础版 train.py 用 argparse 把所有超参扁平地塞进命令行：
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--micro_batch_size", type=int, default=8)
    ...十几个参数平铺，没有分组、没有继承、没法存成文件复用。
本模块用 **dataclass + yaml + CLI 覆盖** 替代：
    - 用 dataclass 定义带类型、带默认值、带文档的配置 schema(对应 verl 的各种 *Config dataclass)
    - 把默认值写进 config.yaml(对应 verl 的 sft_trainer_engine.yaml)
    - 命令行用 key.path=value 覆盖(对应 verl 用的 hydra override 语法 trainer.lr=1e-4)

【这块添加的意义】
1. 结构化: model/data/optim/trainer 分组，一眼看清"这组参数管什么"，改一处不会乱。
2. 可复现: 整份配置可存成 yaml，别人拿到 yaml 就能复现你的实验(配置即实验记录)。
3. 可覆盖: 默认值在 yaml 里，命令行只覆盖要改的，不用每次重写一长串参数。

【和 verl 原版的对比】
verl 用 **hydra + omegaconf**，本模块用纯 dataclass + PyYAML 手写一个最小版。区别:
  - verl 的 hydra 支持 `defaults` 列表组合多个 yaml(如 model@model: hf_model)、
    `${...}` 变量插值、multirun(一次跑多组参数)、@hydra.main 装饰器自动注入。
  - 本最小版只实现"读一份 yaml + key=value 覆盖"，足以理解配置系统的价值；
    想要组合/插值/multirun 时再上 hydra(注释里标了对应关系)。
verl 对应: verl/trainer/config/*.yaml + verl/utils/config.py 的 omega_conf_to_dataclass。
"""
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, get_type_hints

import yaml


# ---------- 各分组的配置 schema(对应 verl 里 *Config dataclass) ----------

@dataclass
class ModelConfig:
    """模型相关配置。对应 verl 的 model: hf_model 块。"""
    path: str = "/home/zzy/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
    trust_remote_code: bool = True
    dtype: str = "bfloat16"  # bf16 / fp32
    # 【新增·模块3】梯度检查点开关。对应 verl model.enable_gradient_checkpointing
    enable_gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """数据相关配置。对应 verl 的 data 块。"""
    train_file: str = "/home/zzy/verl/learning/sft/data/train.parquet"
    test_file: str = "/home/zzy/verl/learning/sft/data/test.parquet"
    max_length: int = 512
    max_train_samples: int = 1000  # -1 表示全部


@dataclass
class OptimConfig:
    """优化器与学习率配置。对应 verl 的 optim 块。"""
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # cosine / constant


@dataclass
class TrainerConfig:
    """训练流程配置。对应 verl 的 trainer 块。"""
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    epochs: int = 2
    seed: int = 42
    bf16: bool = True
    output_dir: str = "outputs/checkpoints/sft-qwen0.5b-gsm8k"
    log_every: int = 5
    save_model: bool = True  # 【新增·模块1】可关掉保存以省磁盘(验证配置时用)
    # 【新增·模块2】实验追踪相关字段。对应 verl trainer.project_name/experiment_name/logger
    project_name: str = "gsm8k-sft"
    experiment_name: str = "qwen0.5b-sft"
    logger: tuple = ("console", "file")  # 后端列表, 对应 verl 的 trainer.logger
    # 【新增·模块4】remove_padding 开关: True=打包(packing, 不右填充), False=右填充(对照)。
    # 对应 verl trainer 里的 remove_padding / collator 配置, verl 默认 True。
    remove_padding: bool = True
    # 【新增·模块5】每个 micro-batch 的 token 预算上限(只在 remove_padding=true 时生效)。
    # 对应 verl trainer.max_token_len_per_gpu。按 token 切 batch -> 各 batch 显存利用率恒定。
    max_token_len_per_gpu: int = 8192
    # 【新增·模块7】分布式进程组配置。对应 verl: verl 不在 config 里存 world_size,
    # 它从 env/torch.distributed 读(verl/utils/distributed.py 直接读 os.environ['WORLD_SIZE'])。
    # 这里 world_size 默认 1 仅作"未起 torchrun 时的占位/日志展示"; 真正的 world_size 在 init 后
    # 用真实值覆盖(见 train.py 的 cfg.trainer.world_size = world_size)。backend 默认 'nccl',
    # 对应 verl 的 get_nccl_backend()。
    world_size: int = 1    # 由 torchrun 注入, train.py 启动后用真实值覆盖; 默认 1=单进程
    backend: str = "nccl"  # 通信后端; 对应 verl f'...:{get_nccl_backend()}' 里的 nccl 段
    # 【新增·模块8】DDP find_unused_parameters: Qwen2 前向所有参数都被用, 默认 False;
    #   若 DDP 报 'unused parameters' / 'Expected to have finished reduction' 再开 True(会变慢)。
    #   对应 verl 引擎配置里的 find_unused_parameters。
    find_unused_parameters: bool = False


@dataclass
class SFTConfig:
    """顶层配置，组合各分组。对应 verl 的 sft_trainer_engine.yaml 顶层结构。"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


# ---------- 从 dict 构造嵌套 dataclass(对应 verl omega_conf_to_dataclass 的最简版) ----------

def _build_dataclass(cls, d: dict):
    """递归地把 dict 填进 dataclass: 字段是 dataclass 就递归，否则直接赋值。"""
    d = dict(d)  # 拷贝，避免改入参
    kwargs = {}
    type_hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name not in d:
            continue  # 用 dataclass 默认值
        val = d[f.name]
        ftype = type_hints.get(f.name, f.type)
        # 嵌套 dataclass 字段: dict -> 递归构造
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _build_dataclass(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


# ---------- CLI 覆盖解析(key.path=value，对应 hydra 的 override 语法) ----------

def _coerce(value_str: str):
    """把命令行字符串转成 python 类型: bool/int/float/None/列表/字符串。"""
    s = value_str.strip()
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("null", "none"):
        return None
    if s.startswith("[") and s.endswith("]"):
        # 简单列表，如 [0.9,0.95]
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x) for x in inner.split(",")]
    # int 优先于 float
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_override(cfg_dict: dict, override: str):
    """把 'data.max_train_samples=200' 这种点路径写进嵌套 dict。"""
    if "=" not in override:
        raise ValueError(f"覆盖参数必须是 key.path=value 形式, 得到: {override}")
    path, val_str = override.split("=", 1)
    keys = path.strip().split(".")
    d = cfg_dict
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = _coerce(val_str)


def load_config(yaml_path: str, overrides=None) -> SFTConfig:
    """读 yaml 默认值 -> 应用 CLI 覆盖 -> 构造 SFTConfig。

    对应 verl: hydra @hydra.main 读 config_name=sft_trainer_engine.yaml，
    再按命令行 override 合并，最后 omega_conf_to_dataclass 转成 dataclass。
    """
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}
    # 把顶层缺失的分组补成空 dict，保证后面都能构造出 dataclass
    for grp in ("model", "data", "optim", "trainer"):
        cfg_dict.setdefault(grp, {})
    for ov in overrides or []:
        _apply_override(cfg_dict, ov)
    return _build_dataclass(SFTConfig, cfg_dict)


def config_to_yaml_str(cfg) -> str:
    """把配置转成 yaml 字符串，方便打印"最终生效的配置"(配置即实验记录)。"""
    def _to_dict(x):
        if dataclasses.is_dataclass(x):
            return {f.name: _to_dict(getattr(x, f.name)) for f in fields(x)}
        if isinstance(x, tuple):
            return list(x)
        return x
    return yaml.dump(_to_dict(cfg), default_flow_style=False, sort_keys=False, allow_unicode=True)
