<div align="center">

# 🚀 PackTron

### **Efficient Data Loader for Large Language Model Training**

[English](README.md) | [简体中文](README.zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)

**Zero Padding | Staged Curriculum | Maximum Training Efficiency | Easy Implementation.**

[Features](#-key-features) • [Why PackTron?](#-why-packtron) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Documentation](#-documentation)

</div>

---

## 🎯 Key Features

### ✨ **Zero Padding Waste**
- **No padding tokens** - All sequences are exactly `sequence_length` through intelligent sentence packing
- **100% token utilization** - Every token in your dataset contributes to training
- **Accurate token counting** - Know exactly how many tokens your model sees during training

### 🎨 **Curriculum Control** 
- **Staged Training Curriculum** - Instantly script the sequence and proportion of datasets to control training focus, for example, using specialized data (like code or math) at specific phases to boost model quality.
- **Enhanced Model Quality** - Implement powerful curriculum learning techniques to improve the model's convergence behavior and final performance.
- **Focused Learning** - Guarantee that the model is exposed to the most relevant, specialized data exactly when needed to optimize the learning curve.

### ⚡ **Production-Grade Performance**
- **Binary storage format** - Pre-tokenized data stored as `.bin`/`.idx` files with memory-mapped I/O
- **C++ accelerated indexing** - Fast sample index building using optimized C++ code (via pybind11)
- **Automatic multi-GPU data distribution** - Automatically shards data across GPUs based on `world_size` for data parallelism (no manual setup needed)
- **Smart epoch management** - Automatically calculates and builds required epochs based on `train_iters`/`eval_iters` and data size, ensuring training never runs out of data

### 🔧 **Easy Integration**
- **Works with `transformers`** - Drop-in replacement for `load_dataset` in your existing training pipelines
- **Simple API** - Three steps: preprocess → config → dataloader
- **Lightweight** - Minimal dependencies, focused on core functionality

---

## 🤔 Why PackTron?

### The Problem with Traditional Data Loading

When training large language models with `datasets`' `load_dataset`, you typically face these issues:

```python
# Traditional approach with transformers
from datasets import load_dataset

dataset = load_dataset("text", data_files="data.jsonl")
# ❌ Requires padding to batch_size × sequence_length
# ❌ Lacks automatic epoch construction
# ❌ Poor flexibility in managing multi-stage data pipelines
# ❌ No automatic data sharding across workers
```

**Problems:**
- 🚫 **Padding waste**: Up to 30-50% of tokens can be padding in typical batches
- 🚫 **Inaccurate metrics**: Token counts include padding, making it hard to track real training progress
- 🚫 **Memory inefficiency**: Padding consumes GPU memory without contributing to learning
- 🚫 **Complex masking**: Need to handle attention masks for padding tokens
- 🚫 **Manual multi-GPU setup**: Requires manual data sharding across GPUs for distributed training
- 🚫 **Data exhaustion risk**: Training may stop unexpectedly if dataset runs out before completing all iterations
- 🚫 **Lack of fine-grained data control**: Cannot easily implement staged training curriculum using different datasets

### The PackTron Solution

PackTron fundamentally solves all these traditional bottlenecks by leveraging the **three-tier data loading architecture** inspired by Megatron-LM:

```python
# PackTron approach
from packtron import create_dataloader, PackTronConfig

config = PackTronConfig(
    train_iters=1000,  # PackTron automatically calculates epochs if data is insufficient
    eval_iters=100,
    train_curriculum=[0.3, 0, 0.2, 1, 0.5, 0] # Setting the usage of different dataset in training
    ...
)
train_loader, eval_loader = create_dataloader(
    tokenizer, config, 
    rank=0,           # Current GPU rank
    world_size=2      # Total GPUs - data automatically distributed!
)
# ✅ All sequences are exactly sequence_length (no padding!)
# ✅ Guaranteed data availability (auto epoch calculation)
# ✅ Flexible configuration of datasets for multi-stage training.
# ✅ Automatic distributed data sharding across GPUs
```

**Benefits:**
- ✅ **Zero padding**: Every sequence is exactly `sequence_length` tokens
- ✅ **Accurate metrics**: Count real training tokens, not padding
- ✅ **Memory efficient**: No wasted memory on padding
- ✅ **Simple**: No attention mask complexity for padding
- ✅ **Automatic multi-GPU support**: Data automatically distributed across GPUs based on `world_size` - just pass `rank` and `world_size` to `create_dataloader`
- ✅ **Guaranteed data availability**: Automatically calculates required epochs and repeats data if needed, ensuring training completes all `train_iters` without interruption
- ✅ **Staged Learning Made Easy**: Control exactly when to use which dataset at runtime for efficient curriculum learning.

---

## 🏗️ How It Works

PackTron uses a **three-layer architecture** based on Megatron-LM's proven data loading design:

### Layer 1: Binary Storage (`IndexedDataset`)
- Pre-processes raw text into tokenized binary format (`.bin`/`.idx` files)
- Uses memory-mapped I/O for efficient random access
- Stores documents as sequences of token IDs

### Layer 2: Sentence Packing (`GPTDataset`)
- Intelligently packs multiple documents into fixed-length sequences
- Uses C++-accelerated indexing for fast sample construction
- Handles document boundaries and sequence alignment

### Layer 3: Dataset Blending (`BlendedDataset`)
- Supports mixing multiple datasets with custom weights
- Handles train/validation splits
- Caches indices for fast subsequent loads

### The Packing Algorithm

PackTron leverages **Megatron-LM's proven sentence packing architecture**, making it accessible to users who don't need the full Megatron framework. The algorithm works as follows:

1. **Document Indexing**: Each document is split into sentences and tokenized
2. **Sequence Construction**: Multiple sentences are concatenated until reaching `sequence_length`
3. **Boundary Handling**: Documents are split across sequences when needed, with proper boundary markers
4. **Index Building**: C++ code builds efficient lookup indices for fast sampling

This ensures:
- Every sequence is exactly `sequence_length` tokens
- No padding is ever needed
- Document boundaries are preserved (with special tokens if needed)

**PackTron's Contribution**: We have streamlined and optimized this powerful data architecture by eliminating Megatron-specific dependencies to provide a clean and easy-to-use API. Beyond this decoupling, we engineered a novel design that offers users unprecedented flexibility to configure data settings, ensuring seamless integration with transformers and other popular frameworks.

---

## 🚀 Quick Start

### Step 1: Preprocess Your Data

Convert your raw text data into PackTron's binary format:

```bash
packtron-preprocess \
    --input data.jsonl \
    --output-prefix data \
    --tokenizer-model t5-base \
    --workers 4 \
    --append-eod
```

This creates binary files:
- `data_text_document.bin` / `data_text_document.idx` - Tokenized data in binary format

**Note**: The output format is `{output_prefix}_{json_key}_{level}.bin/idx`. 
By default, `json_key` is `"text"` and `level` is `"document"` (or `"sentence"` if `--split-sentences` is used).

### Step 2: Configure PackTron

Create a `PackTronConfig` with your training parameters:

```python
from packtron import PackTronConfig

config = PackTronConfig(
    path_to_cache="./cache",           # Cache directory for indices
    split_config="98,2",               # Train:98%, Valid:2%
    data_path="data",                  # Path prefix to .bin/.idx files
    train_iters=1000,                   # Number of training iterations
    eval_iters=100,                     # Total evaluation iterations
    sequence_length=4096,               # Fixed sequence length
    batch_size=4,                       # Batch size per GPU
    random_seed=42,                     # Random seed for reproducibility
    train_curriculum="0.5 0 0.5 1"     # Optional: schedule dataset ids over training (See Step 4)
)
```

### Step 3: Create DataLoader

Use PackTron's `create_dataloader` function:

```python
from packtron import create_dataloader
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create dataloaders (automatically handles multi-GPU)
train_loader, eval_loader = create_dataloader(
    tokenizer=tokenizer,
    config=config,
    rank=0,          # Current GPU rank
    world_size=2,    # Total number of GPUs
    consumed_samples=0
)

# Use in training loop
# PackTron automatically handles:
# - Data distribution across GPUs (each GPU gets its own shard based on rank/world_size)
# - Epoch calculation (repeats data if needed to complete train_iters/eval_iters)
for batch in train_loader:
    tokens = batch['tokens']        # Shape: [batch_size, sequence_length]
    labels = batch['labels']        # Shape: [batch_size, sequence_length]
    attention_mask = batch['attention_mask'].float()  # Shape: [batch_size, sequence_length]
    
    # No padding! All sequences are exactly sequence_length
    outputs = model(input_ids=tokens, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
```

**That's it!** Your data is now efficiently packed with zero padding.

### Step 4 (Optional): Script the Training Dataset Sequence

Need to rotate between coding, math, physics, or other corpora at specific points of training? Set `train_curriculum` in the config:

```python
config = PackTronConfig(
    # ...other fields...
    data_path="0.3 coding_data 0.3 math_data 0.4 physics_data",
    train_curriculum="0.2 0 0.2 1 0.2 2 0.1 0 0.1 1 0.2 2"  # 20% coding (dataset id 0), 20% math, 20% physics, 10% coding, 10% coding, 20% physics
)
```

Based on data_path setting, PackTron automatically allocates 30% of the required data to the coding dataset. If the source data is insufficient, PackTron will automatically pack it into several epochs util enough. Math and physics datasets follow the same logic.

According the train_curriculum, PackTron segments the training data stream based on the curriculum's proportional stages: the first 20% uses coding (Dataset ID 0), followed by 20% of math, then 20% of physics, and so on.

PackTron keeps the underlying Megatron-LM blends cached, then reorders `dataset_index` / `dataset_sample_index` in memory, so you can step through specialized data phases without re-tokenizing or rebuilding `.bin/.idx` files.

---

## 📦 Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.7.0
- C++ compiler (g++ or clang++)
- pybind11 (for C++ extension)

### Install PackTron

```bash
# Clone the repository
git clone https://github.com/JimmyAwoe/PackTron.git
cd PackTron

# Install with pip (recommended)
pip install -e .

# This will:
# - Install all dependencies
# - Automatically compile the C++ extension
```

### Manual Installation

If automatic compilation fails:

```bash
# Install dependencies
pip install -r requirements.txt

# Manually compile C++ extension
cd packtron/utils/
make
cd ../..
```

### Verify Installation

```bash
python -c "from packtron import create_dataloader, PackTronConfig; print('✓ Installation successful!')"
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

---

## 📚 Documentation

### Complete Example

See `examples/llama_train.py` for a complete training example with:
- Multi-GPU distributed training using `torchrun`
- LLaMA model integration
- Evaluation loop
- Checkpoint saving
- Dataset sequencing via `--train-curriculum`

```bash
# Run with 2 GPUs
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

Need a turnkey example? `examples/run.sh` mirrors the command above so you can launch a staged training curriculum with one shell script.

### Curriculum Scheduling Deep Dive

`train_curriculum` accepts a series of `<fraction> <dataset_id>` pairs. PackTron normalizes the fractions, then draws each portion from the requested dataset **without touching the cached indices**. Example interpretation:

```
train_curriculum="0.3 0 0.3 1 0.2 0 0.2 1"
```

- The first 30% of steps use dataset `0`
- The next 30% draw from dataset `1`
- The following 20% return to dataset `0`
- The final 20% complete on dataset `1`

Unlike Hugging Face's `load_dataset`, PackTron doesn't require manual shuffling or repeated preprocessing—curriculum changes are applied instantly at runtime.

### API Reference

#### `PackTronConfig`

Configuration object for PackTron datasets:

```python
@dataclass
class PackTronConfig:
    path_to_cache: str          # Cache directory for dataset indices
    split_config: str           # Data split ratios, e.g., '98,2'
    data_path: str              # Data path prefix(es)
    train_iters: int            # Number of training iterations
    eval_iters: int             # Total evaluation iterations
    sequence_length: int         # Sequence length for training samples
    batch_size: int             # Batch size per GPU
    random_seed: int = 1234     # Random seed for dataset shuffling
    train_curriculum: Optional[str] = None # Optional curriculum schedule
```

#### `create_dataloader`

Factory function to create PackTron DataLoaders:

```python
from packtron import create_dataloader

def create_dataloader(
    tokenizer,                  # Tokenizer instance
    config: PackTronConfig,     # PackTron configuration
    rank: int = 0,              # Current process rank
    world_size: int = 1,        # Total number of processes
    consumed_samples: int = 0   # Samples consumed (for checkpoint resuming)
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Returns (train_dataloader, eval_dataloader)"""
```

#### `build_tokenizer`

Utility function to build tokenizer:

```python
from packtron import build_tokenizer

tokenizer = build_tokenizer(args)  # args should have tokenizer_model attribute
```
The args.tokenizer_model will be sent to 
```python
import transformers

transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.tokenizer_model, **kwargs
        )
```
So you can use any tokenizer that HuggingFace support which achieve complete compatibility between transformers and PackTron.

---

## 🆚 Comparison with Alternatives

| Feature | PackTron | `transformers.load_dataset` | Megatron-LM |
|---------|----------|------------------------------|-------------|
| **Zero Padding** | ✅ Yes | ❌ No | ✅ Yes |
| **Accurate Token Count** | ✅ Yes | ❌ No | ✅ Yes |
| **Easy Integration** | ✅ Simple API | ✅ Simple | ❌ Complex |
| **Lightweight** | ✅ Minimal deps | ✅ Minimal deps | ❌ Heavy |
| **Binary Format** | ✅ Fast I/O | ❌ Text-based | ✅ Fast I/O |
| **C++ Acceleration** | ✅ Yes | ❌ No | ✅ Yes |
| **Auto Multi-GPU Distribution** | ✅ Automatic | ❌ Manual | ✅ Yes |
| **Auto Epoch Management** | ✅ Yes | ❌ No | ✅ Yes |
| **Runtime Dataset Sequencing** | ✅ Reorder without rebuild | ❌ Not supported | ⚠️ Requires custom scripting |

---

## 🎓 Use Cases

PackTron is perfect for:

- 🧪 **Research**: Accurate token counting for reproducible experiments
- 🏭 **Production Training**: Efficient data loading for large-scale model training
- 📊 **Data Analysis**: Understanding exactly what tokens your model sees
- 🚀 **Resource-Constrained Training**: Maximize GPU utilization with zero padding waste

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Note: This project contains code derived from Megatron-LM (NVIDIA), which is licensed under BSD-3-Clause. See [LICENSE](LICENSE) for full license information.

---

## 🙏 Acknowledgments

- **Megatron-LM** - For the excellent data loading architecture
- **transformers** - For the tokenizer integration
- **Facebook Fairseq** - For the original IndexedDataset implementation

---

## 📧 Contact

- **Author**: JimmyAwoe
- **GitHub**: [@JimmyAwoe](https://github.com/JimmyAwoe)
- **Issues**: [GitHub Issues](https://github.com/JimmyAwoe/PackTron/issues)

---

<div align="center">

**Made with ❤️ for the LLM community**

⭐ Star this repo if you find it useful!

</div>

