"""
PackTron - Efficient Sentence Packing for Large Language Model Training

A lightweight library for efficient sentence packing that eliminates padding waste
in LLM training, providing accurate token counting and better training efficiency.
"""

# Version
__version__ = "0.1.0"

# Import main API functions and classes
from .dataloader import (
    create_dataloader,
    build_pretraining_data_loader,
    PackTronDataset,
    PackTronSampler,
)

# Import configuration
from .utils.dataset_config import PackTronConfig

# Import tokenizer utilities
from .utils.tokenizer import build_tokenizer

# Import logging utilities
from .utils.log import log_single_rank

# Import preprocessing script function (if needed)
# Note: preprocess_data.py is typically used as a CLI script

__all__ = [
    "__version__",
    "create_dataloader",
    "build_pretraining_data_loader",
    "PackTronDataset",
    "PackTronSampler",
    "PackTronConfig",
    "build_tokenizer",
    "log_single_rank",
]

