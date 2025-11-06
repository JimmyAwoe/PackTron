<div align="center">

# 🚀 PackTron

### **Efficient Sentence Packing for Large Language Model Training**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)

**Zero Padding. Accurate Token Counting. Maximum Training Efficiency.**

[Features](#-key-features) • [Why PackTron?](#-why-packtron) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Documentation](#-documentation)

</div>

---

## 🎯 Key Features

### ✨ **Zero Padding Waste**
- **No padding tokens** - All sequences are exactly `sequence_length` through intelligent sentence packing
- **100% token utilization** - Every token in your dataset contributes to training
- **Accurate token counting** - Know exactly how many tokens your model sees during training

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

When training large language models with `transformers`' `load_dataset`, you typically face these issues:

```python
# Traditional approach with transformers
from transformers import load_dataset

dataset = load_dataset("text", data_files="data.jsonl")
# ❌ Sequences have variable lengths
# ❌ Requires padding to batch_size × sequence_length
# ❌ Padding tokens waste computation and memory
# ❌ Cannot accurately count training tokens
# ❌ Padding masks add complexity
```

**Problems:**
- 🚫 **Padding waste**: Up to 30-50% of tokens can be padding in typical batches
- 🚫 **Inaccurate metrics**: Token counts include padding, making it hard to track real training progress
- 🚫 **Memory inefficiency**: Padding consumes GPU memory without contributing to learning
- 🚫 **Complex masking**: Need to handle attention masks for padding tokens
- 🚫 **Manual multi-GPU setup**: Requires manual data sharding across GPUs for distributed training
- 🚫 **Data exhaustion risk**: Training may stop unexpectedly if dataset runs out before completing all iterations

### The PackTron Solution

PackTron solves all these problems by **packing multiple documents into fixed-length sequences**:

```python
# PackTron approach
from dataloader import create_dataloader
from utils.dataset_config import PackTronConfig

config = PackTronConfig(
    train_iters=1000,  # PackTron automatically calculates epochs if data is insufficient
    eval_iters=100,
    ...
)
train_loader, eval_loader = create_dataloader(
    tokenizer, config, 
    rank=0,           # Current GPU rank
    world_size=2      # Total GPUs - data automatically distributed!
)
# ✅ All sequences are exactly sequence_length (no padding!)
# ✅ 100% token utilization
# ✅ Accurate token counting
# ✅ Automatic distributed data sharding across GPUs
# ✅ Guaranteed data availability (auto epoch calculation)
```

**Benefits:**
- ✅ **Zero padding**: Every sequence is exactly `sequence_length` tokens
- ✅ **Accurate metrics**: Count real training tokens, not padding
- ✅ **Memory efficient**: No wasted memory on padding
- ✅ **Simple**: No attention mask complexity for padding
- ✅ **Automatic multi-GPU support**: Data automatically distributed across GPUs based on `world_size` - just pass `rank` and `world_size` to `create_dataloader`
- ✅ **Guaranteed data availability**: Automatically calculates required epochs and repeats data if needed, ensuring training completes all `train_iters` without interruption

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
- Handles train/validation/test splits
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

**PackTron's contribution**: We've simplified and lightweighted this powerful architecture, removing Megatron-specific dependencies and providing a clean, easy-to-use API that integrates seamlessly with `transformers` and other popular training frameworks.

---

## 🚀 Quick Start

### Step 1: Preprocess Your Data

Convert your raw text data into PackTron's binary format:

```bash
python preprocess_data.py \
    --input data.jsonl \
    --output-prefix data \
    --tokenizer-model gpt2 \
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
from utils.dataset_config import PackTronConfig

config = PackTronConfig(
    path_to_cache="./cache",           # Cache directory for indices
    split_config="98,2",               # Train:98%, Valid:2%
    data_path="data",                   # Path prefix to .bin/.idx files
    train_iters=1000,                   # Number of training iterations
    eval_iters=100,                     # Total evaluation iterations
    sequence_length=4096,               # Fixed sequence length
    batch_size=4,                       # Batch size per GPU
    random_seed=42                      # Random seed for reproducibility
)
```

### Step 3: Create DataLoader

Use PackTron's `create_dataloader` function:

```python
from dataloader import create_dataloader
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
cd utils/
make
cd ..
```

### Verify Installation

```bash
python -c "from utils.helpers_cpp import build_sample_idx_int32; print('✓ Installation successful!')"
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

---

## 📚 Documentation

### Complete Example

See `example/llama_train.py` for a complete training example with:
- Multi-GPU distributed training using `torchrun`
- LLaMA model integration
- Evaluation loop
- Checkpoint saving

```bash
# Run with 2 GPUs
torchrun --nproc-per-node=2 example/llama_train.py \
    --model-config llama_60m.json \
    --tokenizer-model meta-llama/Llama-2-7b-hf \
    --data-path data \
    --cache-dir ./cache \
    --split-config 98,2 \
    --sequence-length 4096 \
    --batch-size 4 \
    --train-iters 1000 \
    --eval-iters 10 \
    --eval-interval 100
```

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
```

#### `create_dataloader`

Factory function to create PackTron DataLoaders:

```python
def create_dataloader(
    tokenizer,                  # Tokenizer instance
    config: PackTronConfig,     # PackTron configuration
    rank: int = 0,              # Current process rank
    world_size: int = 1,        # Total number of processes
    consumed_samples: int = 0   # Samples consumed (for checkpoint resuming)
) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_dataloader, eval_dataloader)"""
```

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

**Made with ❤️ for the LLM training community**

⭐ Star this repo if you find it useful!

</div>

