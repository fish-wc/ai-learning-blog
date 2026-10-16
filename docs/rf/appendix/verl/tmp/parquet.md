# 自己的理解
将原来按行存取的数据转化为按列存取的方式。因为有的场景需要单独读取列的内容，如果按行读取出来然后再筛选出列，那么存取速度慢了。所以才需要有parquet这个数据格式。




# GSM8K 转 Parquet 完整流程
GSM8K原始是jsonl格式，每条样本就两个字段：
- `question`：数学应用题
- `answer`：带分步推理的解答，**末尾固定用 `#### 数字` 标记最终答案**

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

转Parquet有两种路线：
1. **最简版：直接导出原始question+answer，不做额外处理（看数据、简单SFT）**
2. **RL训练版（VERL/GRPO常用）：预处理，提取ground truth、构建chat prompt，再存parquet**

> Parquet本质就是**把表格（DataFrame / HuggingFace Dataset）直接写入列式二进制文件**。HuggingFace `datasets`库自带`.to_parquet()`，底层pyarrow，自动生成parquet的schema、行组、列块。

## 方式1：最简代码，直接转（只保留原始字段）
### 安装依赖
```bash
pip install datasets pyarrow
```

```python
from datasets import load_dataset

# 自动下载GSM8K
ds = load_dataset("openai/gsm8k", "main")

# 直接保存成parquet，分成train.parquet / test.parquet
ds["train"].to_parquet("gsm8k_train.parquet")
ds["test"].to_parquet("gsm8k_test.parquet")
```

运行后生成：
- `gsm8k_train.parquet`（7473条）
- `gsm8k_test.parquet`（1319条）

读取验证：
```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="gsm8k_train.parquet")
print(ds["train"][0])
```

此时parquet里面只有两列：`question`、`answer`。适合简单SFT、数据分析。

## 方式2：大模型RL训练常用版本（VERL标准预处理）
做GRPO/PPO数学奖励训练时，不能直接用原始字段。
需要：
1. 用正则提取`####`后面的数字作为`ground_truth`
2. 构造chat格式prompt
3. 增加reward_model、metadata等字段
4. 写入parquet

```python
import re
from datasets import load_dataset

def extract_solution(solution_str):
    # 提取 #### 后面的标准答案
    match = re.search(r"####\s*(-?[\d,.]+)", solution_str)
    assert match
    ans = match.group(1).replace(",", "")
    return ans

# 加载数据集
dataset = load_dataset("openai/gsm8k", "main")

def make_process_fn(split):
    def process_fn(example, idx):
        question = example["question"] + " Let's think step by step and output final answer after ####."
        answer_raw = example["answer"]
        gt = extract_solution(answer_raw)
        return {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {"split": split, "index": idx}
        }
    return process_fn

train_ds = dataset["train"].map(make_process_fn("train"), with_indices=True)
test_ds = dataset["test"].map(make_process_fn("test"), with_indices=True)

# 保存parquet
train_ds.to_parquet("gsm8k_train_processed.parquet")
test_ds.to_parquet("gsm8k_test_processed.parquet")
```
> 这个就是VERL框架训练数学模型的标准GSM8K parquet预处理脚本。生成出来的parquet里面包含嵌套字典字段（prompt是列表dict，reward_model是dict），parquet原生支持嵌套结构。

## 底层原理：GSM8K数据如何变成parquet结构
1. 读取jsonl → 加载为**Arrow Table**（HuggingFace Dataset底层就是Arrow）
2. 你map做数据清洗/字段新增，仍然是内存中的表格
3. `.to_parquet()`调用pyarrow，把表格写入parquet二进制文件
    - 每一列单独存储：question列全部文本放一列块，answer放另一列块
    - 自动按行组切分、snappy压缩
    - schema自动写入文件尾部Footer（字段名、类型，包括嵌套结构）

## 读取parquet看schema
```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("gsm8k_train_processed.parquet")
print(pf.schema)
```

## 常见坑
1. 不要用pandas直接处理超长文本，嵌套字典会被转成字符串；**优先用🤗datasets导出parquet**，嵌套结构可以被parquet原生保存。
2. 提取答案正则：注意负数、带逗号的数字，比如`1,234`，要把逗号去掉。
3. 大模型训练时parquet一般分成多个shard（分片parquet文件），避免单个parquet文件过大。

## 对比
- 原始jsonl：行式文本，每次读取需要加载整行（question+answer）
- GSM8K转出来的parquet：列式存储，你可以只读取`prompt`这一列，不需要读其他字段，训练加载更快、IO更小。