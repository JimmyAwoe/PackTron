<div align="center">

<div style="text-align: center; margin: 0; padding: 0;">
    <img src="figures/caption.png" alt="PackTron Logo" style="width: 400px; height: auto; display: block; margin: 0 auto;"/>
</div>

### **用于大规模语言模型训练的高效数据加载器**

[English](README.md) | [简体中文](README.zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)

**零填充 | 细粒度数据配置 | 训练效率 ⬆️⬆️ | 高效集成**

[关键特性](#-关键特性) • [为什么选择 PackTron?](#-为什么选择-packtron) • [快速上手](#-快速上手) • [安装](#-安装) • [文档](#-文档)

</div>

---

## 🎯 关键特性

### ✨ **零填充浪费**

- **序列无填充** - 通过智能句子打包让所有序列都恰好为 `sequence_length`
- **100% Token 利用率** - 数据集中每个 token 都参与训练
- **准确的 Token 计数** - 精确掌握模型在训练中看到的 token 数量

<div style="margin-top: 20px; overflow: hidden;"> 
    <div style="float: left; width: 98%; text-align: center; padding: 0 1%;">
        <img src="figures/load.png" alt="diff load" style="max-width: 98%;"/>
</div>

### 🎨 **细粒度数据配比控制**

- **分阶段训练数据设置** - 灵活编排数据集使用顺序与占比，例如在特定阶段插入代码或数学数据以提升模型质量
- **提升模型表现** - 轻松实现强大的数据配置策略，改进模型收敛和最终效果
- **聚焦学习** - 确保模型在需要的时间点接触到最相关的专业数据

<div style="margin-top: 20px; overflow: hidden;"> 
    <div style="float: left; width: 98%; text-align: center; padding: 0 1%;">
        <img src="figures/advance_function.png" alt="improvement" style="max-width: 80%;"/>
</div>

### ⚡ **生产级性能**

- **二进制存储格式** - 预先分词的数据以 `.bin`/`.idx` 文件存储，支持内存映射 I/O
- **C++ 加速索引** - 通过优化的 C++（pybind11）快速构建样本索引
- **自动多 GPU 数据分发** - 按照 `world_size` 自动切分数据，告别手动配置
- **智能 Epoch 管理** - 根据 `train_iters`/`eval_iters` 与数据规模自动计算并构建所需 epoch，确保训练永不中断

### 🔧 **易于集成**

- **兼容 `transformers`** - 可无缝替换现有训练流程中的 `load_dataset`
- **简单 API** - 三步即可：预处理 → 配置 → 加载器
- **轻量依赖** - 只关注核心能力，安装简单

---

## 🤔 为什么选择 PackTron?

### 传统数据加载的痛点

当你用 `datasets` 的 `load_dataset` 训练大模型时，通常会遇到：

```python
# 基于 transformers 的传统做法
from datasets import load_dataset

dataset = load_dataset("text", data_files="data.jsonl")
# ❌ 需要填充到 batch_size × sequence_length
# ❌ 不能自动构建 epoch
# ❌ 难以管理多阶段数据流水线
# ❌ 无法自动在多 GPU 间分发数据
```

**常见问题：**

- 🚫 **填充浪费**：典型批次中 30-50% 的 token 都是填充
- 🚫 **指标不准确**：token 计数包含 padding token，难以追踪真实训练进度
- 🚫 **内存效率低**：padding token 占用显存却不提供有效学习信号
- 🚫 **掩码复杂**：还需处理填充令牌的注意力掩码
- 🚫 **多 GPU 手动分发**：分布式训练时必须手动切分数据
- 🚫 **数据耗尽风险**：若数据不足，训练可能提前停止,造成训练失败
- 🚫 **缺乏精细控制**：难以实现细粒度的数据控制

### PackTron 的解决方案

PackTron 借鉴 Megatron-LM 的 **三层数据加载架构**，彻底解决上述瓶颈：

```python
# PackTron 使用方式
from packtron import create_dataloader, PackTronConfig

config = PackTronConfig(
    train_iters=1000,  # 数据不足时自动计算 epoch
    eval_iters=100,
    train_curriculum=[0.3, 0, 0.2, 1, 0.5, 0]
    ...
)
train_loader, eval_loader = create_dataloader(
    tokenizer, config,
    rank=0,           # 当前 GPU rank
    world_size=2      # GPU 总数 - 数据自动分发！
)
# ✅ 所有序列长度均为 sequence_length（无填充）
# ✅ 保证训练所需数据量（自动 epoch 计算）
# ✅ 灵活配置不同阶段数据配比
# ✅ 自动处理分布式数据分片
```

**核心优势：**

- ✅ **零填充**：每个序列恰好为 `sequence_length`
- ✅ **真实指标**：只统计有效令牌
- ✅ **显存友好**：避免填充占用 GPU 内存
- ✅ **简洁**：无需编写填充掩码逻辑
- ✅ **自动多 GPU 支持**：仅需传入 `rank` 与 `world_size`
- ✅ **训练不中断**：自动重复数据以完成全部 `train_iters`
- ✅ **高度灵活的数据配置**：运行时即可调整数据顺序，无需重建 `.bin/.idx`

---

## 🏗️ 工作原理

PackTron 采用 Megatron-LM 验证过的 **三层数据加载架构**：

### 第一层：二进制存储 (`IndexedDataset`)

- 将原始文本预处理为分词后的二进制格式（`.bin`/`.idx`）
- 使用内存映射 I/O 实现高效随机访问
- 按令牌 ID 存储文档

### 第二层：句子打包 (`GPTDataset`)

- 将多个文档智能打包到固定长度序列
- 借助 C++ 加速快速构建样本索引
- 处理文档边界并确保序列对齐

### 第三层：数据集混合 (`BlendedDataset`)

- 支持按自定义权重混合多个数据集
- 管理训练/验证划分
- 缓存索引以加速后续加载

### 打包算法

PackTron 使用 **Megatron-LM 成熟的句子打包算法**，同时简化 Megatron 依赖，帮助用户无需接触完整框架即可使用。流程包括：

1. **文档索引**：按句划分文档并进行分词
2. **序列构建**：不断连接句子直至达到 `sequence_length`
3. **边界处理**：必要时在序列间分割文档，并插入特殊标记
4. **索引构建**：C++ 代码生成高效查找索引，便于快速采样

这意味着：

- 每个序列长度都相同
- 完全不需要填充
- 如果需要可保留文档边界

**PackTron 的贡献**：我们在保留核心架构的基础上，剥离 Megatron 相关依赖，提供简洁易用的 API，并额外设计灵活的数据配置机制，与 transformers 等框架完美融合。

---

## 🚀 快速上手

### 第一步：预处理数据

将原始文本转换为 PackTron 的二进制格式：

```bash
packtron-preprocess \
    --input data.jsonl \
    --output-prefix data \
    --tokenizer-model t5-base \
    --workers 4 \
    --append-eod
```

生成的二进制文件包括：

- `data_text_document.bin` / `data_text_document.idx` - 分词后的二进制数据

**说明**：输出格式为 `{output_prefix}_{json_key}_{level}.bin/idx`。默认 `json_key` 为 `"text"`，`level` 为 `"document"`（如果使用 `--split-sentences` 则为 `"sentence"`）。

### 第二步：配置 PackTron

创建 `PackTronConfig`，填写训练参数：

```python
from packtron import PackTronConfig

config = PackTronConfig(
    path_to_cache="./cache",          # 索引缓存目录
    split_config="98,2",              # 训练:98%，验证:2%
    data_path="data",                 # .bin/.idx 文件前缀路径
    train_iters=1000,                 # 训练迭代次数
    eval_iters=100,                   # 验证迭代次数
    sequence_length=4096,             # 固定序列长度
    batch_size=4,                     # 每块 GPU 的 batch size
    random_seed=42,                   # 随机种子
    train_curriculum="0.5 0 0.5 1"    # 可选：数据细粒度配比 （参见第四步）
)
```

### 第三步：创建 DataLoader

使用 PackTron 的 `create_dataloader`：

```python
from packtron import create_dataloader
from transformers import AutoTokenizer

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 构建 DataLoader（自动处理多 GPU）
train_loader, eval_loader = create_dataloader(
    tokenizer=tokenizer,
    config=config,
    rank=0,          # 当前 GPU rank
    world_size=2,    # GPU 总数
    consumed_samples=0
)

# 将其应用在训练循环中
# PackTron 自动处理：
# - 按 rank/world_size 切分数据
# - 计算 epoch，重复数据以完成 train_iters/eval_iters
for batch in train_loader:
    tokens = batch['tokens']        # 形状：[batch_size, sequence_length]
    labels = batch['labels']        # 形状：[batch_size, sequence_length]
    attention_mask = batch['attention_mask'].float()  # 形状：[batch_size, sequence_length]

    # 无需填充！所有序列自动拼接
    outputs = model(input_ids=tokens, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
```

**就是这么简单！** 你的数据现在以零填充方式高效打包。

### 第四步（可选）：编排训练数据配置

想在特定阶段切换编码、数学、物理等语料？在配置中设置 `train_curriculum`：

```python
config = PackTronConfig(
    # ...其他字段...
    data_path="0.3 coding_data 0.3 math_data 0.4 physics_data",
    train_curriculum="0.2 0 0.2 1 0.2 2 0.1 0 0.1 1 0.2 2"
)
```

基于上述 config 的 data_path 设置，PackTron 自动将 30% 的数据需求分配给 coding 数据集，若数据源不足以提供数据需求要求的数据，系统将自动将其打包为多个 epoch，直到足够满足需求。math 和 physics 数据集亦遵循相同的逻辑。

train_curriculum 的设置会要求 PackTron 将训练数据流按比例阶段性划分：前 20% 使用 coding (数据集 ID 0)，随后 20% 使用 math，接着 20% 使用 physics，以此类推。

PackTron 会复用缓存的 Megatron-LM 混合索引，只需在内存中重新排序 `dataset_index` / `dataset_sample_index`，即可在不重新分词或重建 `.bin/.idx` 的情况下切换数据阶段。

---

## 📦 安装

### 先决条件

- Python >= 3.8
- PyTorch >= 2.7.0
- C++ 编译器（g++ 或 clang++）
- pybind11（用于 C++ 扩展）

### 安装 PackTron

```bash
# 克隆仓库
git clone https://github.com/JimmyAwoe/PackTron.git
cd PackTron

# 推荐方式：使用 pip
pip install -e .

# 该命令会：
# - 安装所有依赖
# - 自动编译 C++ 扩展
```

### 手动安装

若自动编译失败，可按以下步骤：

```bash
# 安装依赖
pip install -r requirements.txt

# 手动编译 C++ 扩展
cd packtron/utils/
make
cd ../..
```

### 验证安装

```bash
python -c "from packtron import create_dataloader, PackTronConfig; print('✓ Installation successful!')"
```

更详细的安装说明见 [INSTALL.md](INSTALL.md)。

---

## 📚 文档

### 完整示例

参见 `examples/llama_train.py`，其中包含：

- 使用 `torchrun` 的多 GPU 分布式训练
- 集成 LLaMA 模型
- 验证循环
- 检查点保存
- 通过 `--train-curriculum` 实现数据细粒度配置

```bash
# 使用 2-GPU 运行
torchrun --nproc-per-node=2 examples/llama_train.py \
    --model-config llama_60m.json \
    --tokenizer-model t5-base \
    --data-path data \
    --cache-dir ./cache \
    --split-config 98,2 \
    --sequence-length 4096 \
    --batch-size 4 \
    --train-iters 1000 \
    --eval-iters 10 \
    --eval-interval 100 \
    --train-curriculum 0.3 0 0.3 1 0.2 0 0.2 1
```

想快速体验？执行 `examples/run.sh` 即可复现上述命令，一键启动分阶段训练流程。

### 数据调度深入解析

`train_curriculum` 接受一系列 `<占比> <dataset_id>` 对。PackTron 会将占比归一化，并按顺序从指定数据集中抽样，**无需修改缓存索引**。例如：

```
train_curriculum="0.3 0 0.3 1 0.2 0 0.2 1"
```

解释如下：

- 前 30% 训练步骤使用数据集 `0`
- 接下来的 30% 使用数据集 `1`
- 随后 20% 再回到数据集 `0`
- 最后 20% 使用数据集 `1` 完成训练

与 Hugging Face `load_dataset` 不同，PackTron 无需手动打乱或重复预处理；数据调整可在训练过程中瞬时生效。

### API 参考

#### `PackTronConfig`

PackTron 数据集的配置对象：

```python
@dataclass
class PackTronConfig:
    path_to_cache: str          # 数据集索引缓存目录
    split_config: str           # 数据划分比例，如 '98,2'
    data_path: str              # 数据路径前缀
    train_iters: int            # 训练迭代次数
    eval_iters: int             # 验证迭代次数
    sequence_length: int        # 训练样本序列长度
    batch_size: int             # 每 GPU 的 batch 大小
    random_seed: int = 1234     # 数据集打乱的随机种子
    train_curriculum: Optional[str] = None # 可选的数据调度
```

#### `create_dataloader`

创建 PackTron DataLoader 的工厂函数：

```python
from packtron import create_dataloader

def create_dataloader(
    tokenizer,                  # 分词器实例
    config: PackTronConfig,     # PackTron 配置
    rank: int = 0,              # 当前进程 rank
    world_size: int = 1,        # 总进程数
    consumed_samples: int = 0   # 已消耗样本数（用于断点续训）
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """返回 (train_dataloader, eval_dataloader)"""
```

#### `build_tokenizer`

构建分词器的工具函数：

```python
from packtron import build_tokenizer

tokenizer = build_tokenizer(args)  # args 需包含 tokenizer_model
```

内部调用如下：

```python
import transformers

transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.tokenizer_model, **kwargs
        )
```

因此可使用任意 Hugging Face 支持的分词器，确保 transformers 与 PackTron 完全兼容。

---

## 🆚 与其他方案对比

| 特性                | PackTron            | `datasets.load_dataset` | Megatron-LM     |
| ------------------- | ------------------- | ----------------------- | --------------- |
| **零填充**          | ✅ 支持             | ❌ 不支持               | ✅ 支持         |
| **准确令牌计数**    | ✅ 支持             | ❌ 不支持               | ✅ 支持         |
| **易于集成**        | ✅ API 简洁         | ✅ 简洁                 | ❌ 复杂         |
| **轻量级**          | ✅ 依赖少           | ✅ 依赖少               | ❌ 依赖重       |
| **二进制格式**      | ✅ 快速 I/O         | ❌ 文本格式             | ✅ 快速 I/O     |
| **C++ 加速**        | ✅ 支持             | ❌ 不支持               | ✅ 支持         |
| **自动多 GPU 分发** | ✅ 自动             | ❌ 手动                 | ✅ 支持         |
| **自动 Epoch 管理** | ✅ 支持             | ❌ 不支持               | ✅ 支持         |
| **灵活数据配置**    | ✅ 重新排序无需重建 | ❌ 不支持               | ⚠️ 需自定义脚本 |

---

## 🎓 适用场景

PackTron 非常适合：

- 🧪 **科研**：精确的令牌计数，便于可复现实验
- 🏭 **生产训练**：大规模模型训练的高效数据加载
- 📊 **数据分析**：准确了解模型接触到的令牌
- 🚀 **资源受限训练**：通过零填充浪费最大化 GPU 利用率

---

## 🤝 参与贡献

欢迎贡献代码！可直接提交 Pull Request。

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

注意：项目包含来源于 Megatron-LM（NVIDIA）的代码，其许可证为 BSD-3-Clause。完整信息参见 [LICENSE](LICENSE)。

---

## 🙏 致谢

- **Megatron-LM** - 优秀的数据加载架构
- **transformers** - 分词器集成支持
- **Facebook Fairseq** - IndexedDataset 原始实现

---

## 📧 联系方式

- **作者**：JimmyAwoe
- **GitHub**：[@JimmyAwoe](https://github.com/JimmyAwoe)
- **问题反馈**：[GitHub Issues](https://github.com/JimmyAwoe/PackTron/issues)

---

<div align="center">

**Made with ❤️ for the LLM community**

⭐ 如果你觉得有帮助，欢迎为仓库点个 Star！

</div>
