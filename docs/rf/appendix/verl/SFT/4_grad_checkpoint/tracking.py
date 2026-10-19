# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
日志与实验追踪 —— verl/learning/sft/code/3_tracking/tracking.py

==================== 【新增·模块2】日志与实验追踪 ====================
【和上一版(2_config_system)的区别】
上一版训练时只 print(loss), 跑完终端一关就没了: 没法回看曲线、没法对比多次实验、
没法把指标存成文件事后分析。
本模块抽象一个 Tracking 类(对应 verl/utils/tracking.py 的 Tracking):
  - 统一接口 log(data={metric: value}, step) / finish()
  - 支持多个后端: console(打印) + file(写 jsonl, 可事后查看/画曲线)
  - 训练里把 loss/lr/grad_norm 都通过它记, 不再直接 print

【这块添加的意义】
1. 持久化: 指标写进文件, 训练完也能回看(基础版 print 跑完就丢)。
2. 可对比: 每次实验一个 jsonl, 多次实验并排画曲线就能看差异。
3. 可插拔后端: 现在只接 console+file, 想接 tensorboard/wandb 只需在 Tracking 里加一个分支,
   训练代码一行都不用改 —— 这正是 verl 抽象 Tracking 类的价值。

【和 verl 原版的对比】
verl 的 Tracking 支持 wandb/mlflow/swanlab/tensorboard/clearml/trackio/file/console 等十种后端,
还处理代理、重试、表格日志、生成样本表格等。本最小版只实现 console + file 两个最常用后端,
但接口形状(log/finish)和 verl 完全一致, 加后端的方式也一样(在 __init__ 里挂一个 adapter)。
verl 对应: verl/utils/tracking.py 的 Tracking 类 + FileLogger/_TensorboardAdapter。
"""
import json
import os
from datetime import datetime


class Tracking:
    """统一日志接口。对应 verl/utils/tracking.py 的 Tracking。

    用法:
        tracker = Tracking(project_name, experiment_name, default_backend=["console","file"], config=cfg)
        tracker.log({"train/loss": 0.5, "train/lr": 1e-4}, step=global_step)
        tracker.finish()
    """

    supported_backend = ["console", "file"]  # verl 还支持 wandb/tensorboard/mlflow/...

    def __init__(self, project_name, experiment_name, default_backend=("console",), config=None,
                 output_dir="."):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for b in default_backend:
            assert b in self.supported_backend, f"不支持的后端: {b}"
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.logger = {}
        self._finished = False

        if "console" in default_backend:
            self.logger["console"] = _ConsoleLogger()

        if "file" in default_backend:
            # 对应 verl 的 FileLogger: 写 {output_dir}/{project}/{experiment}.jsonl
            self.logger["file"] = _FileLogger(project_name, experiment_name, output_dir)

    def log(self, data, step, backend=None):
        """记一条指标。data 是 {metric_name: value} 字典, step 是全局步数。"""
        for name, inst in self.logger.items():
            if backend is None or name in backend:
                inst.log(data=data, step=step)

    def finish(self, exit_code=0):
        """收尾: 关闭文件等。幂等。"""
        if self._finished:
            return
        self._finished = True
        for inst in self.logger.values():
            inst.finish()


class _ConsoleLogger:
    """console 后端: 把指标打印到终端。对应 verl 的 LocalLogger(print_to_console=True)。"""

    def log(self, data, step):
        parts = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"[step {step}] {parts}")

    def finish(self):
        pass


class _FileLogger:
    """file 后端: 每条指标写一行 jsonl, 事后可读出画曲线。对应 verl 的 FileLogger。

    格式(每行): {"step": N, "ts": "...", "data": {metric: value, ...}}
    文件路径: {output_dir}/{project_name}/{experiment_name}.jsonl
    """

    def __init__(self, project_name, experiment_name, output_dir="."):
        directory = os.path.join(output_dir, project_name)
        os.makedirs(directory, exist_ok=True)
        self.filepath = os.path.join(directory, f"{experiment_name}.jsonl")
        print(f"[Tracking] 文件日志: {os.path.abspath(self.filepath)}")
        # buffering=0 立即落盘, 训练中途断电也能保住已写的
        self.fp = open(self.filepath, "w", buffering=1)

    def log(self, data, step):
        record = {
            "step": step,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "data": {k: float(v) if isinstance(v, (int, float)) else v for k, v in data.items()},
        }
        self.fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self):
        self.fp.close()
