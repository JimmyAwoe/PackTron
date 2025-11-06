"""
Dataset Configuration Module
"""
from dataclasses import dataclass

@dataclass
class PackTronConfig:
    """Configuration for PackTron dataset and training"""
    
    path_to_cache: str 
    """Directory path for caching dataset indices"""
    
    split_config: str
    """Data split ratios, e.g., '98,2' for train:98%, valid:2%"""
    
    data_path: str
    """Data path prefix(es), can be a single path or weighted paths like '0.5 path1 0.5 path2'"""
    
    train_iters: int
    """Number of training iterations"""
    
    eval_iters: int
    """Total number of evaluation iterations across all evaluations"""
    
    sequence_length: int 
    """Sequence length for training samples"""
    
    batch_size: int
    """Batch size per GPU"""
    
    random_seed: int = 1234
    """Random seed for dataset shuffling""" 
    


